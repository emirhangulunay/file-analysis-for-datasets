import scipy.io


class MatFileReader:
    def __init__(self, path_way: str):
        self.path_way = path_way

    def choosed_file_reader(self):
        mat = scipy.io.loadmat(self.path_way)
        print(mat)
