
class XY_metadata :

    def __init__(self, root_data:str, X_filename :str, Y_filename :str, X_join:[], Y_join:[], target_col_name:str):
        self.root_data = root_data
        self.X_filename = X_filename
        self.Y_filename = Y_filename
        self.X_join = X_join
        self.Y_join = Y_join
        self.target_col_name = target_col_name

    def is_training_set(self) -> bool :
        return True if self.target_col_name is not None else False

    def is_XY_in_separate_file(self) -> bool:
        return True if self.Y_filename is not None else False