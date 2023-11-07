# AlphaToe


### fig 1 goals ~ 1.5 pages including the figure

- don't know nothin'
- have info
- have graphs
- done


1. It has a world model folks.
    a. Turn taking in attention heads - Have info
    b. Understands win conditions - don't know nothin'
        a. Each move has associated positions in the residual stream - know somethin'
            a. Subtract residual stream cases where certain move is present from cases where it's not present
            b. .95 of the variance is accounted for by 8 dimensions of the residual stream pre-mlp
            c. What is the PCA of the output of the individual attention heads? Are they writing to the same places?
        b. Understand the MLP with sparse autoencoders - know somethin'
        c. Can we trick the attention heads
            0,1,2
            mlp sees on move 1, move 3, and move 5
            can we feed the head 0,1,2 and 1,3,5 
            what if we feed an attention head an out of order sequence
    c. Understands legal moves - have info
        a. We have evals showing functional behavior - have info
        b. after 9 moves game is always over (w/ nice equation) - have info
        c. uses components of content embedding to know when to not repeat moves - have info


We have a lot of information about how much subspace overlap the heads and mlps have with each other. Content and positional.
    - every head overlaps with the content embeddings around 0.4
    - ^ .25

- sub figure for each one of these points ^ 
- associated discussion and proof
- table paired with figure 1.a that shows our evals and how well it does

We could do 1.a, 1.c, and 1.d without breaking a sweat. 
The real issue is 1.b

At the end of figure 1 they should have an idea of what we build, the evals, and how it has a world model.
We repeated many of the things from the Othello paper

residual stream: seq x d_model
mlp: d_model x d_mlp x d_mlp x d_model