import tkinter as tk

from gui2_application import Application

def main():
    root = tk.Tk()
    app = Application(root)
    app.pack()
    app.mainloop()

if __name__ == '__main__':
    main()