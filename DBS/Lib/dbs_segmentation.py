import slicer, vtk
import numpy as np
from scipy.ndimage import label as cc_label, find_objects as cc_find_objects, binary_closing
from ModelRegistration import ModelRegistrationLogic

def _get_or_create_segmentation(volumeNode, baseName="Segmentation"):
    """Return a segmentation node named `baseName` if present; otherwise create it.
    Always ensures reference geometry and display nodes exist."""
    try:
        segNode = slicer.util.getNode(baseName)
        if segNode.GetClassName() != "vtkMRMLSegmentationNode":
            # If a different node has that name, make a new proper one
            raise slicer.util.MRMLNodeNotFoundException("")
    except slicer.util.MRMLNodeNotFoundException:
        segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", baseName)
    segNode.CreateDefaultDisplayNodes()
    segNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
    return segNode

def _ensure_target_segment(segmentationNode, preferredName="tube"):
    """Return a segmentId to work with:
       - if there are segments, use the last one (latest added)
       - otherwise create an empty segment (named `preferredName`) and use it
    """
    seg = segmentationNode.GetSegmentation()
    n = seg.GetNumberOfSegments()
    if n == 0:
        # Create one and grab its id
        seg.AddEmptySegment(preferredName)
        segmentId = seg.GetNthSegmentID(seg.GetNumberOfSegments() - 1)
    else:
        # Use the most recently added segment
        segmentId = seg.GetNthSegmentID(n - 1)
    return segmentId

def _numpy_from_segment(segmentationNode, segmentId, referenceVolumeNode):
    """Export the given segment to a temporary LabelMapVolume and return a NumPy mask [Z,Y,X]."""
    # Make sure binary labelmap representation exists
    binaryName = slicer.vtkSegmentationConverter.GetBinaryLabelmapRepresentationName()
    segmentationNode.GetSegmentation().CreateRepresentation(binaryName)

    # Export to a labelmap aligned with the volume geometry
    labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
        segmentationNode, [segmentId], labelmapNode, referenceVolumeNode
    )
    arr = slicer.util.arrayFromVolume(labelmapNode)  # [Z,Y,X]
    slicer.mrmlScene.RemoveNode(labelmapNode)
    if arr is None:
        # Guard against empty/failed export
        arr = np.zeros(slicer.util.arrayFromVolume(referenceVolumeNode).shape, dtype=np.uint8)
    return (arr > 0).astype(np.uint8)





def run_seg_pipeline(path_or_volumeNode, target, stl_path):
    # ---------------------------
    # INPUT HANDLING
    # ---------------------------
    if isinstance(path_or_volumeNode, str):
        volumeNode = slicer.util.loadVolume(path_or_volumeNode)
        if volumeNode is None:
            raise RuntimeError(f"Failed to load volume from: {path_or_volumeNode}")
    else:
        volumeNode = path_or_volumeNode
        if volumeNode is None:
            raise ValueError("No input volume node provided.")

    # ============================================================
    # ================ PART 1 (threshold segmentation) ===========
    # ============================================================
    segmentationNode = _get_or_create_segmentation(volumeNode, baseName="Segmentation")
    segmentId = _ensure_target_segment(segmentationNode, preferredName="tube")

    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentEditorWidget.setSegmentationNode(segmentationNode)
    segmentEditorWidget.setSourceVolumeNode(volumeNode)
    segmentEditorWidget.setCurrentSegmentID(segmentId)

    segmentEditorWidget.setActiveEffectByName("Threshold")
    effect = segmentEditorWidget.activeEffect()

    ###### testing automatic thresholding ######
    # Convert image data to numpy array
    imageData = slicer.util.arrayFromVolume(volumeNode)
    flattened = imageData.flatten()

    # Compute percentile-based thresholds
    lower_percentile = np.percentile(flattened, 99.2)
    upper_percentile = np.percentile(flattened, 99.9)

    print(f"Auto-threshold range: {lower_percentile:.2f} to {upper_percentile:.2f}")
    effect.setParameter("MinimumThreshold", lower_percentile)
    effect.setParameter("MaximumThreshold", upper_percentile)

    effect.self().onApply()

    segmentEditorWidget = None
    slicer.mrmlScene.RemoveNode(segmentEditorNode)

    segmentationNode.CreateClosedSurfaceRepresentation()

    # ============================================================
    # ============ PART 2 (remove unnecessary parts) =============
    # ============================================================

    seg_np = _numpy_from_segment(segmentationNode, segmentId, volumeNode)

    labeled_array, _ = cc_label(seg_np)
    object_slices = cc_find_objects(labeled_array)

    ASPECT_RATIO_THRESHOLD = 4
    tube_labels = []
    for i, object_slice in enumerate(object_slices):
        if object_slice is None:
            continue
        z_slice, y_slice, x_slice = object_slice
        height = y_slice.stop - y_slice.start
        width  = x_slice.stop - x_slice.start
        aspect_ratio = (height / width) if width > 0 else height * 10
        if aspect_ratio > ASPECT_RATIO_THRESHOLD:
            tube_labels.append(i + 1)

    final_tube_mask = np.isin(labeled_array, tube_labels).astype(np.uint8)

    slicer.util.updateSegmentBinaryLabelmapFromArray(
        final_tube_mask, segmentationNode, segmentId, volumeNode
    )
    segmentationNode.CreateClosedSurfaceRepresentation()

    # ============================================================
    # =============== PART 3 (smoothing + bbox fill)  ============
    # ============================================================
    ## 09/18/2025 testing
    
    # bounding boxes (sagittal + coronal)
    final_mask = np.zeros_like(final_tube_mask, dtype=bool)

    # Z-planes
    y_range_z = float('-inf')
    y_range_slice = slice(None)
    for z in range(final_tube_mask.shape[0]):
        current_slice = final_tube_mask[z, :, :]
        coords = np.argwhere(current_slice)
        if coords.size:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            y_range_current = y_max - y_min
            if y_range_current > y_range_z:
                y_range_z = y_range_current
                y_range_slice = slice(y_min, y_max + 1)
            final_mask[z, y_min:y_max+1, x_min:x_max+1] = True
    
    # 09/23/2025
    for z in range(final_tube_mask.shape[0]):
        current_slice = final_tube_mask[z, :, :]
        coords = np.argwhere(current_slice)
        if coords.size:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            final_mask[z, y_range_slice, x_min:x_max+1] = True

    # X-columns
    y_range_x = float('-inf')
    y_range_slice = slice(None)
    for x in range(final_tube_mask.shape[2]):
        current_slice = final_tube_mask[:, :, x]
        coords = np.argwhere(current_slice)
        if coords.size:
            z_min, y_min = coords.min(axis=0)
            z_max, y_max = coords.max(axis=0)
            y_range_current = y_max - y_min
            if y_range_current > y_range_x:
                y_range_x = y_range_current
                y_range_slice = slice(y_min, y_max + 1)
            final_mask[z_min:z_max+1, y_min:y_max+1, x] = True

    for x in range(final_tube_mask.shape[2]):
        current_slice = final_tube_mask[:, :, x]
        coords = np.argwhere(current_slice)
        if coords.size:
            z_min, y_min = coords.min(axis=0)
            z_max, y_max = coords.max(axis=0)
            final_mask[z_min:z_max+1, y_range_slice, x] = True

    print("Updating the segment with the bounding box result...")
    slicer.util.updateSegmentBinaryLabelmapFromArray(
        final_mask.astype("uint8"),
        segmentationNode, segmentId, volumeNode
    )
    segmentationNode.CreateClosedSurfaceRepresentation()
    print("Segmentation successfully refined with bounding boxes\n")

    # ============================================================
    # ======= PART 3.5 (Model Registration) - 09/23/2025 =========
    # ============================================================

    print("Converting segmentation to model for registration...")
    segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
    segmentation_model_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', "TubeSegmentationModel")
    slicer.modules.segmentations.logic().ExportSegmentToRepresentationNode(segment, segmentation_model_node)
    print(f"Successfully created model: {segmentation_model_node.GetName()}")

    # Load the source model (the ideal tube shape)
    # moving_model_node = slicer.util.loadModel('tube-Body001.stl')
    moving_model_node = slicer.util.loadModel(stl_path) 

    output_transform_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "RegistrationTransform")

    # Run Registration
    reg_logic = ModelRegistrationLogic()
    success = reg_logic.run(
        inputSourceModel=moving_model_node,
        inputTargetModel=segmentation_model_node,
        outputSourceToTargetTransform=output_transform_node,
        transformType=0,  
        numIterations=200
    )

    if success:
        print("Registration completed successfully!")
        moving_model_node.SetAndObserveTransformNodeID(output_transform_node.GetID())
        mean_distance = reg_logic.ComputeMeanDistance(moving_model_node, segmentation_model_node, output_transform_node)
        print(f"Mean distance after registration: {mean_distance:.4f}")
    else:
        print("Registration failed.")
        return None, float('inf')


    # ============================================================
    # =============== PART 4 (Centerline and Projection) ===============
    # ============================================================

    print("Extracting centerline from registered model...")
    point_coordinates = slicer.util.arrayFromModelPoints(moving_model_node)

    bottom_points = point_coordinates[point_coordinates[:, 2] == 0]
    top_points    = point_coordinates[point_coordinates[:, 2] == 45]

    bottom_center_local = bottom_points.mean(axis=0)
    top_center_local    = top_points.mean(axis=0)

    # Create a line node with the local coordinates
    lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "CenterLine")
    lineNode.AddControlPointWorld(vtk.vtkVector3d(*bottom_center_local))
    lineNode.AddControlPointWorld(vtk.vtkVector3d(*top_center_local))

    # Apply the registration transform to the line to move it into world space
    lineNode.SetAndObserveTransformNodeID(output_transform_node.GetID())

    lineNode.HardenTransform()
    
    disp = lineNode.GetDisplayNode()
    disp.SetColor(1, 0, 1)  
    disp.SetLineWidth(2.0)
    disp.SetGlyphScale(1.5)

    print(f"Successfully created centerline from registered model.")

    # Clean up the scene for better visualization
    segmentation_model_node.GetDisplayNode().SetVisibility(0)
    segmentationNode.GetDisplayNode().SetVisibility(0)
    moving_model_node.GetDisplayNode().SetVisibility2D(True)

    ### PROJECTION
    print("Projecting target point onto the new centerline...")

    centerline_endpoints_ras = np.array([
        lineNode.GetNthControlPointPositionVector(0), 
        lineNode.GetNthControlPointPositionVector(1)  
    ])

    centroid = centerline_endpoints_ras.mean(axis=0)
    direction = centerline_endpoints_ras[1] - centerline_endpoints_ras[0]
    

    # Find the parametric equation:
    # Compute centroid
    #arr = np.array(centerline_points_ras)
    #centroid = arr.mean(axis=0)

    # Subtract centroid
    #X = arr - centroid

    # Singular Value Decomposition (PCA)
    #_, _, vh = np.linalg.svd(X)
    #direction = vh[0]  # first principal component

    print("Point on line (centroid):", centroid)
    print("Direction vector:", direction)
    print(f"Line equation: r(t) = {centroid} + t * {direction}")

    ras = np.array([0.0, 0.0, 0.0])
    target.GetNthControlPointPosition(0,ras)

    z_target = ras[2]
    t = (z_target - centroid[2]) / direction[2]
    point_at_z = centroid + t * direction
    target.AddControlPointWorld(point_at_z,"projected")

    #z_target = ras[2]+30
    #t = (z_target - centroid[2]) / direction[2]
    #point_at_z = centroid + t * direction
    #target.AddControlPointWorld(point_at_z,"projected")
    #print(ras[0:1])
    v1 = np.array([ras[0],ras[1]])
    v2 = np.array([point_at_z[0],point_at_z[1]])
    #calculate the error:
    print(ras)
    error = np.linalg.norm(v1 - v2)
    print(error)

    results = {
        "volumeNode": volumeNode,
        "segmentationNode": segmentationNode,
        "segmentationModelNode": segmentation_model_node,
        "registeredModelNode": moving_model_node,
        "transformNode": output_transform_node,
        "centerlineNode": lineNode,
        "final_mask": final_mask
    }

    return results,error

