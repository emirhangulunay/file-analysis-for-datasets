from scripts import process_management

class StartApp:
    def welcome(self):
        while True:
            answer = input("Are you want to read file?\n(Y/N or enter anything character)->").strip()
            if answer == "":
                continue
            if answer.lower() == "n":
                break
            process_management.ProcessManagement.start()

            