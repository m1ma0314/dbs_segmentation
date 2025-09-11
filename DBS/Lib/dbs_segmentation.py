import slicer, vtk
import numpy as np
from scipy.ndimage import label as cc_label, find_objects as cc_find_objects, binary_closing

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

def run_seg_pipeline(path_or_volumeNode,target, *, do_centerline=False):
    """
    Fully automatic pipeline. Reuses the last segmentation & last segment so you never
    depend on a hard-coded name, and multiple Apply clicks work cleanly.
    """
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

    # Segment Editor: Threshold into *that* segment
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentEditorWidget.setSegmentationNode(segmentationNode)
    segmentEditorWidget.setSourceVolumeNode(volumeNode)
    segmentEditorWidget.setCurrentSegmentID(segmentId)

    segmentEditorWidget.setActiveEffectByName("Threshold")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("MinimumThreshold", "40")
    effect.setParameter("MaximumThreshold", "125")
    effect.self().onApply()

    # Clean up editor widget/node
    segmentEditorWidget = None
    slicer.mrmlScene.RemoveNode(segmentEditorNode)

    # Optional: show surface
    segmentationNode.CreateClosedSurfaceRepresentation()

    # ============================================================
    # ============ PART 2 (remove unnecessary parts) =============
    # ============================================================
    # Robust read of the segment mask as NumPy
    seg_np = _numpy_from_segment(segmentationNode, segmentId, volumeNode)

    labeled_array, _ = cc_label(seg_np)
    object_slices = cc_find_objects(labeled_array)

    ASPECT_RATIO_THRESHOLD = 3.5
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
    print("Smoothing the segmentation...")
    smoothed_mask = binary_closing(final_tube_mask.astype(bool), iterations=4)

    print("Updating the segment with the smoothed version...")
    slicer.util.updateSegmentBinaryLabelmapFromArray(
        smoothed_mask.astype("uint8"),
        segmentationNode, segmentId, volumeNode
    )
    segmentationNode.CreateClosedSurfaceRepresentation()
    print("Segmentation successfully updated with the smoothed version!")

    # bounding boxes (sagittal + coronal)
    final_mask = np.zeros_like(smoothed_mask, dtype=bool)

    # Z-planes
    for z in range(smoothed_mask.shape[0]):
        current_slice = smoothed_mask[z, :, :]
        coords = np.argwhere(current_slice)
        if coords.size:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            final_mask[z, y_min:y_max+1, x_min:x_max+1] = True

    # X-columns
    for x in range(smoothed_mask.shape[2]):
        current_slice = smoothed_mask[:, :, x]
        coords = np.argwhere(current_slice)
        if coords.size:
            z_min, y_min = coords.min(axis=0)
            z_max, y_max = coords.max(axis=0)
            final_mask[z_min:z_max+1, y_min:y_max+1, x] = True

    print("Updating the segment with the bounding box result...")
    slicer.util.updateSegmentBinaryLabelmapFromArray(
        final_mask.astype("uint8"),
        segmentationNode, segmentId, volumeNode
    )
    segmentationNode.CreateClosedSurfaceRepresentation()
    print("Segmentation successfully updated")

    results = {
        "volumeNode": volumeNode,
        "segmentationNode": segmentationNode,
        "segmentId": segmentId,          # <-- use this id everywhere; no names needed
        "final_mask": final_mask,
        "smoothed_mask": smoothed_mask,
    }

    # ============================================================
    # =============== PART 4 (OPTIONAL centerline) ===============
    # ============================================================
    if do_centerline:
        print("Calculating centerline directly from slice midpoints...")
        centerline_points = []
        for y in range(smoothed_mask.shape[1]):
            current_slice = final_mask[:, y, :]
            coords = np.argwhere(current_slice)
            if coords.size:
                mid_z, mid_x = coords.mean(axis=0)
                centerline_points.append((mid_z, y, mid_x))

        ijkToRasMatrix = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRasMatrix)

        centerline_points_ras = []
        for ijk_point in centerline_points:
            p_ijk = [ijk_point[2], ijk_point[1], ijk_point[0], 1]  # I,J,K -> X,Y,Z
            p_ras = [0, 0, 0, 0]
            ijkToRasMatrix.MultiplyPoint(p_ijk, p_ras)
            centerline_points_ras.append(p_ras[0:3])

        curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        curveNode.SetName("CenterLine")
        for point in centerline_points_ras:
            curveNode.AddControlPoint(vtk.vtkVector3d(point[0], point[1], point[2]))
        dn = curveNode.GetDisplayNode()
        if dn:
            dn.SetColor(1, 0, 1)
            dn.SetGlyphScale(1.0)
            dn.SetLineWidth(1.0)

        print(f"Successfully created a line with {len(centerline_points_ras)} points.")
        results["curveNode"] = curveNode


        # Find the parametric equation:
        # Compute centroid
        arr = np.array(centerline_points_ras)
        centroid = arr.mean(axis=0)

        # Subtract centroid
        X = arr - centroid

        # Singular Value Decomposition (PCA)
        _, _, vh = np.linalg.svd(X)
        direction = vh[0]  # first principal component

        print("Point on line (centroid):", centroid)
        print("Direction vector:", direction)
        print(f"Line equation: r(t) = {centroid} + t * {direction}")

        ras = [0,0,0]
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


    return results,error

