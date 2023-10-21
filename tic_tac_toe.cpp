#include <iostream>
#include <vector>

// Function to draw the Tic Tac Toe board
void drawBoard(const std::vector<char>& board) {
    std::cout << " " << board[0] << " | " << board[1] << " | " << board[2] << "\n";
    std::cout << "-----------\n";
    std::cout << " " << board[3] << " | " << board[4] << " | " << board[5] << "\n";
    std::cout << "-----------\n";
    std::cout << " " << board[6] << " | " << board[7] << " | " << board[8] << "\n";
}

// Function to check if the game is over
bool isGameOver(const std::vector<char>& board) {
    // Check rows, columns, and diagonals
    for (int i = 0; i < 3; i++) {
        if (board[i] == board[i + 3] && board[i] == board[i + 6] && board[i] != ' ')
            return true;
        if (board[3 * i] == board[3 * i + 1] && board[3 * i] == board[3 * i + 2] && board[3 * i] != ' ')
            return true;
    }

    if (board[0] == board[4] && board[0] == board[8] && board[0] != ' ')
        return true;
    if (board[2] == board[4] && board[2] == board[6] && board[2] != ' ')
        return true;

    // Check for a tie
    for (int i = 0; i < 9; i++) {
        if (board[i] == ' ')
            return false;
    }
    return true;
}

int main() {
    std::vector<char> board(9, ' '); // Initialize an empty board
    char currentPlayer = 'X';
    int move;

    std::cout << "Tic Tac Toe Game\n";

    while (true) {
        drawBoard(board);

        // Get player's move
        std::cout << "Player " << currentPlayer << ", enter your move (1-9): ";
        std::cin >> move;

        if (move < 1 || move > 9 || board[move - 1] != ' ') {
            std::cout << "Invalid move. Try again.\n";
            continue;
        }

        // Update the board
        board[move - 1] = currentPlayer;

        // Check if the game is over
        if (isGameOver(board)) {
            drawBoard(board);
            if (currentPlayer == 'X')
                std::cout << "Player X wins!\n";
            else
                std::cout << "Player O wins!\n";
            break;
        }

        // Switch players
        if (currentPlayer == 'X')
            currentPlayer = 'O';
        else
            currentPlayer = 'X';
    }

    return 0;
}
