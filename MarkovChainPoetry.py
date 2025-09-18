# markov_chains.py
"""Uses Markov chains to write poetry."""

import argparse
import numpy as np

class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        A (numpy.ndarray): The transition matrix containing probabilities
        for transitioning between states.
        states (list): The list of state labels for the transition matrix.
        labels (dict): A dictionary mapping each state label to its
        corresponding index in the transition matrix.
    """

    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]]), states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        # Check that the columns of the provided transition matrix sum to 1.
        if np.allclose(A.sum(axis=0), np.ones(A.shape[1])) == False:
            raise ValueError("A must be a stochastic matrix.")
        # Store the transition matrix A and the state labels as attributes
        self.A = A
        self.states = states

        if states is not None:
            # If state labels are provided, store them in a numbered dictionary.
            self.labels = {state: i for i, state in enumerate(states)}
        else:
            # Otherwise, make one label for each column of the transition matrix.
            self.labels = {i: i for i in range(A.shape[0])}


    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        # Get the outgoing probabilities for the current state
        probabilities = self.A[:, self.labels[state]]
        # Draw a random sample based on the given probabilities
        sample = np.random.multinomial(1, probabilities)
        # Find the index of the state to transition to
        index = np.argmax(sample)
        # Return the next state
        return self.states[index]


    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        # Start the list containing the path from start to stop
        path = [start]
        currentstate = start

        # Make random transitions until the the stop label is reached, saving each transition
        while currentstate != stop:
            currentstate = self.transition(currentstate)
            path.append(currentstate)
        
        # Return the path created
        return path

# Inherits from MarkovChain class
class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        A (numpy.ndarray): The transition matrix containing the probability
        that a certain word is followed by another specific word.
        states (list): The list of unique words in the file.
        labels (dict): A dictionary mapping each unique word to its
        corresponding index in the transition matrix.
    """
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents.
        """
        # Save the contents of the file
        file = open(filename, "r")
        trainingdata = file.read()
        file.close()
        # Separate the string into lines
        sentences = trainingdata.split('\n')
        # Separate the string into words and remove duplicate entries
        words = list(set(trainingdata.split()))
        # Insert start and stop tokens at the beginning and end of list of unique words
        words.insert(0, "$tart")
        words.append("$top")
        # Update states and labels according to training data
        self.states = words
        self.labels = {state: i for i, state in enumerate(words)}
        # Make a transition matrix sized according to the number of unique words
        transitionmatrix = np.zeros((len(words), len(words)))

        # For each line in the input
        for sentence in sentences:
            # Split each line into words
            wordsinsentence = sentence.split()
            # Insert start and stop tokens at the beginning and end of the line
            wordsinsentence.insert(0, "$tart")
            wordsinsentence.append("$top")
            for i in range(len(wordsinsentence) - 1):
                # Add 1 tp the place in the transition matrix corresponding to transitioning from the word to the next word
                transitionmatrix[self.labels[wordsinsentence[i + 1]]][self.labels[wordsinsentence[i]]] += 1
        
        # Make the transition matrix stochastic and save it
        transitionmatrix[-1][-1] += 1
        transitionmatrix /= transitionmatrix.sum(axis = 0)
        self.A = transitionmatrix


    def write(self, num_lines=20):
        """Create a random sentence using MarkovChain.path().

        Parameters:
            lines (int): The number of lines of poetry to write (default 20).

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> poem = SentenceGenerator("poem.txt")
            >>> print(poem.write(1))
            Once upon a midnight dreary, while I pondered, weak and weary,
            Over many a quaint and curious volume of forgotten lore.
        """
        # Write the specified number of lines and return the string
        output = ""
        for i in range(num_lines):
            output += " ".join(self.path("$tart", "$top")[1:-1]) + "\n"
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate poetry or song lyrics using a Markov chain."
    )
    parser.add_argument("filename", help="Path to the input text file")
    parser.add_argument(
        "-n", "--num_lines", type=int, default=20,
        help="Number of lines to generate (default: 20)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Optional path to save the generated poem. If not provided, prints to terminal."
    )
    args = parser.parse_args()

    # Build the generator from the input file
    generator = SentenceGenerator(args.filename)

    # Generate the poem
    poem = generator.write(args.num_lines)

    # Handle output
    if args.output:
        with open(args.output, "w") as f:
            f.write(poem)
        print(f"Poem saved to {args.output}")
    else:
        print(poem)