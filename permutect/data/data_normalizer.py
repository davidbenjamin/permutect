import torch

MAPPING_QUALITY_COLUMN_INDEX = 0
BASE_QUALITY_COLUMN_INDEX = 1
MAX_MAP_QUALITY = 60
MAX_BASE_QUALITY = 30
DEFAULT_CAP = 1e4


class DataNormalizer:
    '''
    Basically, this knows what to do with every column.  It stores a dict from int (column index) to float, float tuple.
    The normalization is subtract the first element and divide by the second.
    '''
    def __init__(self, is_read: bool):
        self.cap_shift_scale_by_column = {}
        self.is_read = is_read

    def fit(self, array2d):
        tensor2d = torch.from_numpy(array2d)
        for col_idx in range(tensor2d.shape[1]):
            column = tensor2d[:, col_idx]
            float_column = column.float()

            # cases
            # 1) binary column  -- do nothing
            # 2) mapping quality -- almost all values between 0 and 60, and 30-60 much more common than 0 to 30.
            #    in this case identify the max MQ as some high percentile and divide by this max MQ
            # 3) majority is zero -- do nothing
            # 4) otherwise, median and IQR
            if self.is_read and col_idx == MAPPING_QUALITY_COLUMN_INDEX:    # THIS IS MAPPING QUALITY!!! hard-code to cap at 60 and scale by 60
                self.cap_shift_scale_by_column[col_idx] = (MAX_MAP_QUALITY, 0, MAX_MAP_QUALITY)
            elif self.is_read and col_idx == BASE_QUALITY_COLUMN_INDEX:    # THIS IS BASE QUALITY!!! hard-code to cap at 30 and scale by 30
                self.cap_shift_scale_by_column[col_idx] = (MAX_BASE_QUALITY, 0, MAX_BASE_QUALITY)
            elif torch.mean(torch.logical_or(column == 0, column == 1).float()) > 0.999999:     # binary column
                self.cap_shift_scale_by_column[col_idx] = (DEFAULT_CAP, 0, 1)    # identity transform
            elif torch.mean((column == 0).float()) > 0.6:     # mainly-zero
                self.cap_shift_scale_by_column[col_idx] = (DEFAULT_CAP, 0, 1)    # identity transform
            else:
                iqr = (torch.quantile(float_column, q=0.75) - torch.quantile(float_column, q=0.25)).item()
                self.cap_shift_scale_by_column[col_idx] = (DEFAULT_CAP, torch.median(float_column).item(), iqr)

        # TODO: turn this check on if reads, but not for INFO of course
        # assert num_mq_columns == 1, f"Did not find exactly one MQ column, found {num_mq_columns}."

    def transform(self, array2d):
        result = torch.from_numpy(array2d)
        for col_idx in range(result.shape[1]):
            cap, shift, scale = self.cap_shift_scale_by_column[col_idx]
            result[:, col_idx] = (torch.clamp(result[:, col_idx], max=cap) - shift) / scale
        return result.numpy()

