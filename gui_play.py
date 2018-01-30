#import board
#import pieces
import chess
import bot
import Tkinter as tk
from PIL import Image, ImageTk


class BoardGuiTk(tk.Frame):
    pieces = {}
    selected = None
    selected_piece = -1
    hilighted = []
    icons = {}

    color1 = "white"
    color2 = "grey"

    rows = 8
    columns = 8

    @property
    def canvas_size(self):
        return (self.columns * self.square_size, self.rows * self.square_size)

    def __init__(self, parent, chessboard, square_size=128):

        self.chessboard = chessboard
        self.square_size = square_size
        self.parent = parent

        canvas_width = self.columns * square_size
        canvas_height = self.rows * square_size

        tk.Frame.__init__(self, parent)

        self.canvas = tk.Canvas(
            self, width=canvas_width, height=canvas_height, background="grey")
        self.canvas.pack(side="top", fill="both", anchor="c", expand=True)

        self.canvas.bind("<Configure>", self.refresh)
        self.canvas.bind("<Button-1>", self.click)

        self.statusbar = tk.Frame(self, height=64)
        self.button_quit = tk.Button(
            self.statusbar, text="New", fg="black", command=self.reset)
        self.button_quit.pack(side=tk.LEFT, in_=self.statusbar)

        self.label_status = tk.Label(
            self.statusbar, text="   White's turn  ", fg="black")
        self.label_status.pack(side=tk.LEFT, expand=0, in_=self.statusbar)

        self.button_quit = tk.Button(
            self.statusbar,
            text="Quit",
            fg="black",
            command=self.parent.destroy)
        self.button_quit.pack(side=tk.RIGHT, in_=self.statusbar)
        self.statusbar.pack(expand=False, fill="x", side='bottom')

    def click(self, event):

        if self.chessboard.is_game_over() or self.chessboard.turn == chess.BLACK:
            return
        # Figure out which square we've clicked
        col_size = row_size = event.widget.master.square_size

        current_column = event.x / col_size
        current_row = 7 - (event.y / row_size)

        position = current_row * 8 + current_column

        print("click at %d-%d" % (current_row, current_column))
        if self.selected_piece != -1:
            print("selected piece %s" % (str(self.selected_piece)))

            did_move = self.move(self.selected_piece, position)

            self.selected_piece = -1
            self.pieces = {}
            if did_move:
                self.hilighted = []
            else:
                self.hilighted.pop()

            self.refresh()
            self.draw_pieces()
            self.refresh()
            if did_move and not self.chessboard.is_game_over():
                self.parent.after(200, self.next_move)

        else:
            self.hilighted = []
            self.hilight(position)
            self.draw_pieces()
            self.refresh()

    def next_move(self):

        if self.chessboard.is_game_over():
            return
        self.bot_move()

    def bot_move(self):
        move = bot.get_move(self.chessboard)
        self.hilighted = [move.from_square, move.to_square]
        self.chessboard.push(move)
        self.draw_pieces()
        self.refresh()

    def move(self, p1, p2):
        piece = self.chessboard.piece_at(p1)
        move = chess.Move(p1, p2)
        if piece.symbol().lower() == "p" and (p2 / 8 == 0 or p2 / 8 == 7):
            print("promoting to queen")
            move.promotion = chess.QUEEN
        if self.chessboard.is_legal(move):
            self.chessboard.push(move)
            self.label_status["text"] = " " + ("white" if piece.color else
                                               "black") + ": " + str(move)
            return True
        else:
            self.label_status["text"] = "BAD"

            return False

    def hilight(self, pos):
        piece = self.chessboard.piece_at(pos)
        print("hilight piece: %s" % str(piece))
        if piece is not None and (piece.color == self.chessboard.turn):
            self.selected_piece = pos
            self.hilighted.append(pos)
        else:
            print("bad piece")

    def addpiece(self, name, image, row=0, column=0):
        '''Add a piece to the playing board'''
        self.canvas.create_image(
            0, 0, image=image, tags=(name, "piece"), anchor="c")
        self.placepiece(name, row, column)

    def placepiece(self, name, row, column):
        '''Place a piece at the given row/column'''
        self.pieces[name] = (row, column)
        x0 = (column * self.square_size) + int(self.square_size / 2)
        y0 = ((7 - row) * self.square_size) + int(self.square_size / 2)
        self.canvas.coords(name, x0, y0)

    def refresh(self, event={}):
        '''Redraw the board'''
        if event:
            xsize = int((event.width - 1) / self.columns)
            ysize = int((event.height - 1) / self.rows)
            self.square_size = min(xsize, ysize)

        self.canvas.delete("square")
        color = self.color2
        for row in range(self.rows):
            color = self.color1 if color == self.color2 else self.color2
            for col in range(self.columns):
                x1 = (col * self.square_size)
                y1 = ((7 - row) * self.square_size)
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                if (self.selected is not None) and (row, col) == self.selected:
                    self.canvas.create_rectangle(
                        x1,
                        y1,
                        x2,
                        y2,
                        outline="black",
                        fill="orange",
                        tags="square")
                elif (row * 8 + col in self.hilighted):
                    self.canvas.create_rectangle(
                        x1,
                        y1,
                        x2,
                        y2,
                        outline="black",
                        fill="spring green",
                        tags="square")
                else:
                    self.canvas.create_rectangle(
                        x1,
                        y1,
                        x2,
                        y2,
                        outline="black",
                        fill=color,
                        tags="square")
                color = self.color1 if color == self.color2 else self.color2
        for name in self.pieces:
            self.placepiece(name, self.pieces[name][0], self.pieces[name][1])
        self.canvas.tag_raise("piece")
        self.canvas.tag_lower("square")

    def draw_pieces(self):
        self.canvas.delete("piece")
        for square_index, piece in self.chessboard.piece_map().iteritems():
            x = square_index / 8
            y = square_index % 8
            if piece is not None:
                filename = "img/%s%s.png" % ("white"
                                             if piece.color else "black",
                                             piece.symbol().lower())
                piecename = "%s%s%s" % (piece.symbol(), x, y)

                if (filename not in self.icons):
                    self.icons[filename] = ImageTk.PhotoImage(
                        file=filename, width=32, height=32)

                self.addpiece(piecename, self.icons[filename], x, y)
                self.placepiece(piecename, x, y)

    def reset(self):
        self.chessboard.load(board.FEN_STARTING)
        self.refresh()
        self.draw_pieces()
        self.refresh()


def display(chessboard):
    root = tk.Tk()
    root.title("Simple Python Chess")

    gui = BoardGuiTk(root, chessboard)
    gui.pack(side="top", fill="both", expand="true", padx=4, pady=4)
    gui.draw_pieces()

    #root.resizable(0,0)
    root.mainloop()


if __name__ == "__main__":
    display(chess.Board())
