from scripts import start_app


class Main:
    def __init__(self):
        start_app.StartApp().welcome()

    @staticmethod
    def start():
        Main()

if __name__ == "__main__":
    Main.start()