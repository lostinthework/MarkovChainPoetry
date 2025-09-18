# MarkovChainPoetry
A Markov-chain–based poetry generator that creates poems or song lyrics from example text input.

I made this project as a fun way to apply linear algebra and probability theory concepts that I learned in my college classes at BYU.

To use this program, include a filepath argument to a text file containing at least one poem or the lyrics to at least one song. For best resuts, only include songs or poems that sound similar to each other, i. e. written by the same author. Optionally include -n to specify how many lines long you would like your poem to be, or -o to specify an output file to write to. By default, 20 lines of poetry will be written and printed to the terminal.

Example usage:

python MarkovChainPoetry.py input.txt -n 15 -o poem.txt

(This will read the poetry in 'input.txt' and write a 15 line poem to 'poem.txt'.)
