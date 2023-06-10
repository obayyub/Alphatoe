import pytest
from src.game import Board, State

class TestBoard():
    def test_get_possible_moves(self):
        board = Board()
        assert len(board.get_possible_moves()) == 9
        board.make_move(1)
        assert len(board.get_possible_moves()) == 8

    def test_make_move(self):
        board = Board()
        for i in range(9):  
            board.make_move(i)
        assert (board.game_state == State.DRAW) or (board.game_state == State.OVER)
        with pytest.raises(ValueError):
            board.make_move(0)          
        




        
        
    

