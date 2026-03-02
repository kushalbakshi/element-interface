import os
import h5py
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix


class EXTRACT_loader:
    def __init__(self, extract_file_path: str):
        """Initialize EXTRACT loader class
        Args:
            extract_file_path (str): string, absolute file path to EXTRACT output file.
        """
        self.creation_time = datetime.fromtimestamp(os.stat(extract_file_path).st_ctime)

        try:
            results = loadmat(extract_file_path)
            self.S = results["output"][0]["spatial_weights"][0]
            self.spatial_weights = self.S.transpose([2, 0, 1])
            self.T = results["output"][0]["temporal_weights"][0]
        except NotImplementedError:
            # v7.3 mat file - use h5py
            results = h5py.File(extract_file_path, "r")

            spatial_weights_grp = results["output"]["spatial_weights"]

            # Check if this is our converted ndsparse format
            if "type" in spatial_weights_grp and "sparse_2d" in spatial_weights_grp:
                # Converted ndsparse format
                self.spatial_weights = self._load_converted_ndsparse(
                    spatial_weights_grp
                )
            else:
                # Original format
                self.spatial_weights = spatial_weights_grp[:]

            self.T = results["output"]["temporal_weights"][:]

    def _load_converted_ndsparse(self, grp):
        """Load spatial weights from converted ndsparse struct.

        Args:
            grp: h5py group containing 'type', 'nd_shape', and 'sparse_2d'

        Returns:
            scipy.sparse.csc_matrix in (n_cells, height, width) arrangement,
            or reconstructed as list of 2D sparse arrays per cell
        """
        # Read the shape - stored as column vector, so flatten
        nd_shape = grp["nd_shape"][:].flatten().astype(int)
        height, width, n_cells = nd_shape

        # Load the 2D sparse matrix from HDF5
        # MATLAB sparse is stored as: data, ir (row indices), jc (column pointers)
        sparse_grp = grp["sparse_2d"]
        data = sparse_grp["data"][:]
        ir = sparse_grp["ir"][:].astype(int)  # Row indices (0-indexed in HDF5)
        jc = sparse_grp["jc"][:].astype(int)  # Column pointers

        # Reconstruct as scipy CSC sparse matrix
        # Shape is (height*width, n_cells)
        sparse_2d = csc_matrix((data, ir, jc), shape=(height * width, n_cells))

        # Convert to (n_cells, height, width) format
        # Each column becomes a 2D image for one cell
        # Store as list of sparse 2D arrays for memory efficiency
        spatial_weights = []
        for i in range(n_cells):
            col = sparse_2d.getcol(i).toarray().flatten()
            # Reshape to 2D - MATLAB uses column-major (Fortran) order
            cell_mask = col.reshape((height, width), order="F")
            spatial_weights.append(csc_matrix(cell_mask))

        return spatial_weights

    def load_results(self):
        """Load the EXTRACT results
        Returns:
            masks (dict): Details of the masks identified with the EXTRACT segmentation package.
        """
        from scipy.sparse import find, issparse

        masks = []
        for mask_id, s in enumerate(self.spatial_weights):
            # Handle both sparse and dense arrays
            if issparse(s):
                ypixels, xpixels, weights = find(s)
            else:
                ypixels, xpixels, weights = find(csc_matrix(s))

            masks.append(
                dict(
                    mask_id=mask_id,
                    mask_npix=len(weights),
                    mask_weights=weights,
                    mask_center_x=int(np.average(xpixels, weights=weights) + 0.5),
                    mask_center_y=int(np.average(ypixels, weights=weights) + 0.5),
                    mask_center_z=None,
                    mask_xpix=xpixels,
                    mask_ypix=ypixels,
                    mask_zpix=None,
                )
            )
        return masks
