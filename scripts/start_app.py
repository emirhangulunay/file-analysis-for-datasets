from scripts import process_management

class StartApp:
    def __init__(self):
        self.check = False

    def welcome(self):
        while True:
            self.check = bool(input("Are you want to read file?\n(y/n or enter anything character)->"))
            if self.check == True:
                process_management.ProcessManagement.start()
                self.check = False

            elif self.check == "n" or self.check == "N": break
            else: break

            