""" orchestras the whole operation,
    basically should ask and trace polymarket action 24/7 via the
    @247_data_poly, when the market active (and even from an hour before
    because im pretty sure i can take data from the after hours or before training
    i just cant trade) open \ close positions depending on the signals for
    each position opening (check the DB constantly) the main thing
    is it should work asynchronically and manage the requests to the db
    and stuff like that """