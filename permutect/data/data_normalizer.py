import torch


class DataNormalizer:
    '''
    Basically, this knows what to do with every column.  It stores a dict from int (column index) to float, float tuple.
    The normalization is subtract the first element and divide by the second.
    '''
    def __init__(self):
        self.shift_and_scale_by_column = {}

    def fit(self, array2d):
        tensor2d = torch.from_numpy(array2d)
        num_mq_columns = 0
        for col_idx in range(tensor2d.shape[1]):
            column = tensor2d[:, col_idx]
            float_column = column.float()

            # cases
            # 1) binary column  -- do nothing
            # 2) mapping quality -- almost all values between 0 and 60, and 30-60 much more common than 0 to 30.
            #    in this case identify the max MQ as some high percentile and divide by this max MQ
            # 3) majority is zero -- do nothing
            # 4) otherwise, median and IQR
            if torch.mean(torch.logical_or(column == 0, column == 1).float()) > 0.999999:     # binary column
                self.shift_and_scale_by_column[col_idx] = (0, 1)    # identity transform
            elif torch.mean(torch.logical_and(column > -1, column < 61).float()) > 0.99 and torch.median(float_column) > 40:
                num_mq_columns += 1
                max_mq = torch.quantile(float_column, q=0.99).item()
                self.shift_and_scale_by_column[col_idx] = (0, max_mq)
            elif torch.mean((column == 0).float()) > 0.6:     # mainly-zero
                self.shift_and_scale_by_column[col_idx] = (0, 1)    # identity transform
            else:
                iqr = (torch.quantile(float_column, q=0.75) - torch.quantile(float_column, q=0.25)).item()
                self.shift_and_scale_by_column[col_idx] = (torch.median(float_column).item(), iqr)

        # TODO: turn this check on if reads, but not for INFO of course
        # assert num_mq_columns == 1, f"Did not find exactly one MQ column, found {num_mq_columns}."

    def transform(self, array2d):
        result = torch.from_numpy(array2d)
        for col_idx in range(result.shape[1]):
            shift, scale = self.shift_and_scale_by_column[col_idx]
            result[:, col_idx] = (result[:, col_idx] - shift) / scale
        return result.numpy()

