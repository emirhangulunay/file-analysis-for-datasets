import os
import platform
from pathlib import Path
from scripts.operations import mat_file_reader as mfr
from scripts.operations import hea_file_reader as hfr

class DirectoryLister:
    def __init__(self):
        os.chdir("/" if platform.system()=="Linux" else "C:/")
        self.base = os.getcwd()
        self.exec_path_dict = {}

    def show_path_folders(self):
        self.exec_path_dict = {i: os.path.join(self.base, item) 
                               for i, item in enumerate(os.listdir(self.base)) }

        for i, e in self.exec_path_dict.items(): print(f"{i} <- {e}")
        return self.choose_number()


class PathManager(DirectoryLister):
    def __init__(self, file_ext):
        super().__init__()
        self.file_ext = file_ext
        self.choosedPathNumber = 0

    def choose_number(self):
        try:
            self.choosedPathNumber = int(input("Choose index -> "))
            if  self.choosedPathNumber not in self.exec_path_dict.keys(): raise IndexError 
            return self.open_path()

        except ValueError as e:
            print(f"{e} please choose integer")
            return self.choose_number()
        
        except IndexError as i:
            print(f"{i} please enter true index")
            return self.choose_number()

    def open_path(self):
        choosed_path = Path(self.exec_path_dict[self.choosedPathNumber])

        if choosed_path.is_file():
            if choosed_path.suffix == self.file_ext:
                if self.file_ext == ".mat":
                    reader = mfr.MatFileReader(str(choosed_path))
                    reader.choosed_file_reader()
                elif self.file_ext == ".hea":
                    reader = hfr.HeaFileReader(str(choosed_path))
                    reader.choosed_file_reader()
                else:
                    print(f"No reader registered for: {self.file_ext}")
            else:
                print(f"Selected file is not a {self.file_ext} file.")
            return None

        if choosed_path.is_dir():
            os.chdir(choosed_path)
            self.base = os.getcwd()
            return self.show_path_folders()

        print("Selected path is neither file nor directory.")
        return None
    
    @classmethod
    def start(cls):
        return cls().show_path_folders()
    



