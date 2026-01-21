from scripts import path_management
import os


class ProcessManagement:
    def __init__(self):
        self.choosed_operations = 0
        self.operations_dict = dict(enumerate(self._list_operations()))

    def _list_operations(self):
        ops_dir = "/home/emirhan/Desktop/dataset_analysis/scripts/operations"
        entries = []
        for name in os.listdir(ops_dir):
            if not name.endswith(".py"):
                continue
            if name.startswith("__"):
                continue
            entries.append(name)
        entries.sort()
        return entries

    def show_operations(self):
        for i, e in self.operations_dict.items(): 
            print(f"{i} - {e}")
        return self.choose_operations()

    def choose_operations(self):
        try:
            self.choosed_operations = int(input("Choosed operation ->"))

            if self.choosed_operations not in self.operations_dict.keys():
                raise IndexError

            if self.operations_dict[self.choosed_operations] == "mat_file_reader.py":
                return self.mat_file()

            if self.operations_dict[self.choosed_operations] == "hea_file_reader.py":
                return self.hea_file()

            print("Selected operation is not supported.")
            return self.show_operations()

        except ValueError as e:
            print(f"{e} please choose integer")
            return self.choose_operations()

        except IndexError as i:
            print(f"{i} please enter true index")
            return self.choose_operations()

        except Exception as exc:
            print(f"Unexpected error: {exc}")
            return self.show_operations()

    def mat_file(self):
        path_management.PathManager.start(".mat")

    def hea_file(self):
        path_management.PathManager.start(".hea")

    @classmethod
    def start(cls):
        return cls().show_operations()

