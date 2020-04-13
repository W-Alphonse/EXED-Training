import os


class XY_metadata :

    def __init__(self,  X_pathname :[], Y_pathname :[], X_join:[], Y_join:[], target_col_name:str):
        self.X_pathname = os.path.join(X_pathname[0], *X_pathname[1:])
        self.Y_pathname = None if Y_pathname is None else os.path.join(Y_pathname[0], *Y_pathname[1:])
        self.X_join = X_join
        self.Y_join = Y_join
        self.target_col_name = target_col_name

    def is_training_set(self) -> bool :
        return True if self.target_col_name is not None else False

    def is_XY_in_separate_file(self) -> bool:
        return True if self.Y_pathname is not None else False