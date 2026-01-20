from scripts import path_management
import os


class ProcessManagement:
    def __init__(self):
        self.choosed_operations = 0
        self.operations_dict = dict(
            enumerate(
                [i for i in os.listdir("/home/emirhan/Desktop/dataset_analysis/scripts/operations")]
            )
        )

    def show_operations(self):
        for i, e in self.operations_dict.items(): 
            print(f"{i} - {e}")
        return self.choose_operations()

    def choose_operations(self):
        try:
            self.choosed_operations = int(input("Choosed operation ->"))

            if self.choosed_operations not in self.operations_dict.keys():
                raise IndexError

            # operations
            if self.operations_dict[self.choosed_operations] == "mat_file_reader.py":
                return self.mat_file()

            return self.open_path()

        except ValueError as e:
            print(f"{e} please choose integer")
            return self.choose_operations()

        except IndexError as i:
            print(f"{i} please enter true index")
            return self.choose_operations()

    def mat_file(self):
        path_management.PathManager.start(".mat")

    @classmethod
    def start(cls):
        return cls().show_operations()

