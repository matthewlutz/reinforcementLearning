//Matthew Lutz CS457, 12/09/2024

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Main {
	
	enum action{ UP, DOWN, LEFT, RIGHT} //enum to map all possible actions(not including drift)
	private static Map<State, Map<action, Double>> qTable = new HashMap<>();

	public static void main(String[] args) {
		String filename = null; //filename
		double alpha = .9; //learning rate
		double randomness = .9; //randomness
		double discountRate = .9; //discount rate
		int learningDecay = 1000; //decay of the learning rate
		int randomnessDecay = 200; //decay of the randomness
		double successProbability = .8; //action succes propability
		boolean qFlag = false; //toggles the q-learning
		int trials = 10000; //number of trials
		boolean unicode = false; //toggles the use of unicode characters
		int verbosity = 1; //verbosity
		

		
		try {
			for(int i = 0; i < args.length; i++) {
				switch(args[i]) {
					case "-f":
	                    if (++i < args.length) {
	                  	  filename = args[i];
	                      //System.out.println(filename);
	                    }
	                    else throw new IllegalArgumentException("Expected filename after -f");
	                    break;
					case "-a":
						if (++i < args.length) {
							alpha = Double.parseDouble(args[i]);
		                }
						else throw new IllegalArgumentException("Expected filename after -f");
		                    break;	                
					case "-e":
	                    if (++i < args.length) {
	                    	  randomness = Double.parseDouble(args[i]);
	                      }
	                      else throw new IllegalArgumentException("Expected alpha after -a");
	                      break;
					case "-g":
	                    if (++i < args.length) {
	                  	  discountRate = Double.parseDouble(args[i]);
	                    }
	                    else throw new IllegalArgumentException("Expected epochLimit after -e");
	                    break;
					case "-na":
	                    if (++i < args.length) {
	                  	  learningDecay = Integer.parseInt(args[i]);
	                    }
	                    else throw new IllegalArgumentException("Expected batchsize after -m");
	                    break;     
					case "-ne":
	                    if (++i < args.length) {
	                  	  randomnessDecay = Integer.parseInt(args[i]);
	                    }
	                    else throw new IllegalArgumentException("Expected lambda after -l");
	                    break; 
					case "-p":
						if (++i < args.length) {
							successProbability = Double.parseDouble(args[i]);
		                    }
		                    else throw new IllegalArgumentException("Expected epochLimit after -e");
		                    break;
					case "-q":
	                    qFlag = true;
	                    break;
					case "-T":
	                    if (++i < args.length) {
	                  	  trials = Integer.parseInt(args[i]);
	                    }
	                    else throw new IllegalArgumentException("Expected weightinit after -w");
	                    break;
					case "-u":
	                    unicode = true;
	                    break;
					case "-v":
	                    if (++i < args.length) {
	                  	  verbosity = Integer.parseInt(args[i]);
	                    }
	                    else throw new IllegalArgumentException("Expected verbosity after -v");
	                    break;
				}
			}
			
		}catch (NumberFormatException e) {
	        System.err.println("Error parsing numerical argument: " + e.getMessage());
	        return;
	    } catch (IllegalArgumentException e) {
	        System.err.println("Argument error: " + e.getMessage());
	        return;
	    }
	
	    if (filename == null) {
	        System.err.println("Filename must be specified with -f.");
	        return;
	    }
	    
	    if(filename != null) {
			List<List<Character>> grid = readDataFromFile(filename);
			System.out.println("Read file");
			
			initializeQTable(grid, qTable);
			//System.out.println("Q-Table initialized with " + qTable.size() + " states.");
			System.out.println("* Beginning " + trials + " learning episodes with SARSA...");
			System.out.println("  Done with learning!");
			//train the agent 
			trainAgent(grid, trials, alpha, discountRate, randomness, successProbability, learningDecay, randomnessDecay, qFlag, verbosity);
			System.out.println("* Beginnning 50 evaluation episodes...");
			double avgReward = evaluatePolicy(qTable, grid, 50, successProbability);
			System.out.println("  Avg. Total Reward of Learned Policy: " + avgReward);
			System.out.println("* Learned greedy policy:");
			printPolicy(grid, unicode);
			
			if (verbosity >= 2) {
			    System.out.println("* Learned Q values:");
			    printQValues(grid);
			}
			
	    }    
	}
	
	//this method takes advantage of a 2d list to store the characters from the input file
	public static List<List<Character>> readDataFromFile(String filename){
		List<List<Character>> grid = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue; 
                }
                List<Character> row = new ArrayList<>();
                for (char c : line.toCharArray()) {
                    row.add(c);
                }
                grid.add(row);
            }
        } catch(FileNotFoundException e) {
	        System.err.println("File not found: " + filename);
	    } catch (IOException e) {
	        System.err.println("Error reading file: " + filename);
	    }
        return grid;
    }
	
	//initiialize all possible action values to 0.0
	private static void initializeQTable(List<List<Character>> grid, Map<State, Map<action, Double>> qTable) {
		for(int i =0; i < grid.size(); i++) {
			for(int j =0; j < grid.get(i).size(); j++) {
				char currentCell = grid.get(i).get(j);
				if(currentCell == 'B' || currentCell == 'M') {
					//mine cell or blocked 
					continue;
				}
				
				State currState = new State(i, j);
	            Map<action, Double> actionValues = new HashMap<>();
	            
	            for(action a : action.values()) {
	            	actionValues.put(a, 0.0);
	            }
	            
	            qTable.put(currState, actionValues);
			}
		}
	}
	
	
	//helper method to get the reward for a state
	private static double getReward(char cell) {
		switch(cell) {
			case 'G': return 10.0;
			case 'M': return -100.0;
			case 'C': return -20.0;
			case '_': return -1.0;
			case 'S': return 0.0;
			default: return 0.0;
		}
	}
	
	
	//helper method to find the current start state
	public static State findStart(List<List<Character>> grid) {
		for(int i =0; i < grid.size(); i++){
			for(int j =0; j < grid.get(i).size(); j++) {
				if(grid.get(i).get(j) == 'S') {
					return new State(i,j);
				}
			}
		}
	    throw new IllegalStateException("Start state (S) not found in the grid.");
	}
	
	
	//helper method to help with agents movement. Success probability is the chance that it does not drift
	public static State performAction(State currState, action movement, double successProbability, List<List<Character>> grid) {
		int row = currState.row;
	    int col = currState.column;

	    //intended target cell
	    int intendedRow = row;
	    int intendedCol = col;
	    switch (movement) {
	        case UP: intendedRow--; break;
	        case DOWN: intendedRow++; break;
	        case LEFT: intendedCol--; break;
	        case RIGHT: intendedCol++; break;
	    }

	    //4 possible drift options
	    List<State> driftOptions = new ArrayList<>();
	    switch (movement) {
	        case UP: 
	            if (isValidMove(row, col - 1, grid)) driftOptions.add(new State(row, col - 1)); 
	            if (isValidMove(row, col + 1, grid)) driftOptions.add(new State(row, col + 1)); 
	            break;
	        case DOWN: 
	            if (isValidMove(row, col - 1, grid)) driftOptions.add(new State(row, col - 1)); 
	            if (isValidMove(row, col + 1, grid)) driftOptions.add(new State(row, col + 1)); 
	            break;
	        case LEFT: 
	            if (isValidMove(row - 1, col, grid)) driftOptions.add(new State(row - 1, col));
	            if (isValidMove(row + 1, col, grid)) driftOptions.add(new State(row + 1, col));
	            break;
	        case RIGHT: 
	            if (isValidMove(row - 1, col, grid)) driftOptions.add(new State(row - 1, col)); 
	            if (isValidMove(row + 1, col, grid)) driftOptions.add(new State(row + 1, col)); 
	            break;
	    }

	    // use a random number to compare against the success probability
	    double random = Math.random();
	    if (random < successProbability) {
	        if (isValidMove(intendedRow, intendedCol, grid)) {
	            return new State(intendedRow, intendedCol);
	        }
	    } else if (!driftOptions.isEmpty()) {
	        //random drift option
	        int driftIndex = (random < successProbability + (1 - successProbability) / 2) ? 0 : 1;
	        if (driftIndex < driftOptions.size()) {
	            return driftOptions.get(driftIndex);
	        }
	    }
	    
	    //stay in same place if movement fails
	    return currState;
		
	}
	
	//helper method to ensure a agents move is valid. 
	private static boolean isValidMove(int row, int col, List<List<Character>> grid) {
	    return row >= 0 && row < grid.size() && col >= 0 && col < grid.get(0).size() && grid.get(row).get(col) != 'B';
	}
	
	
	//method that trains the agent
	private static void trainAgent(List<List<Character>> grid, int episodes, double initialAlpha, double gamma, double initialEpsilon, double successProbability, int learningDecay, int randomDecay, boolean qFlag, int verbosity) {
	    State startState = findStart(grid);
	    double alpha = initialAlpha;
	    double epsilon = initialEpsilon;
	    if(verbosity >= 3) {
	    	System.out.println("* Episode        Current Greedy Policy");
	    }

	    for (int episode = 0; episode < episodes; episode++) {
	        State currState = startState;
	        action currentAction = selectAction(currState, epsilon);

	        for (int step = 0; step < grid.size() * grid.get(0).size(); step++) { 
	            State nextState = performAction(currState, currentAction, successProbability, grid);
	            char cellType = grid.get(nextState.row).get(nextState.column);
	            double reward = getReward(cellType);

	            //entered a terminal state, update qtable and break
	            if (cellType == 'G' || cellType == 'M') {
	                qTable.get(currState).put(currentAction, qTable.get(currState).get(currentAction) + alpha * (reward - qTable.get(currState).get(currentAction)));
	                break;
	            }

	            action nextAction = selectAction(nextState, epsilon);

	            //update q-value
	            double currentQ = qTable.get(currState).get(currentAction);
	            double targetQ = qFlag
	                ? qTable.get(nextState).get(nextAction) //on policy
	                : qTable.get(nextState).values().stream().max(Double::compare).orElse(0.0); //off policy

	            qTable.get(currState).put(currentAction, currentQ + alpha * (reward + gamma * targetQ - currentQ));

	            currState = nextState;
	            currentAction = nextAction;
	        }
	        
	        //alpha decay
	        if ((episode + 1) % learningDecay == 0) {
	            alpha = initialAlpha / (1 + (episode / learningDecay));
	            if (verbosity >= 4) {
	                System.out.printf("    (after episode %d, alpha to %.5f)\n", episode + 1, alpha);
	            }
	        }
	        //random decay
	        if ((episode + 1) % randomDecay == 0) {
	            epsilon = initialEpsilon / (1 + (episode / randomDecay));
	            if (verbosity >= 4) {
	                System.out.printf("    (after episode %d, epsilon to %.5f)\n", episode + 1, epsilon);
	            }
	        }

	        int evalInterval = episodes/10;
	        if (verbosity >= 3 && (episode + 1) % evalInterval == 0) {
	            double avgReward = evaluatePolicy(qTable, grid, 50, successProbability);
	            System.out.printf("     %-12d %.3f\n", episode + 1, avgReward);
	            continue;
	        }	       	       
	    }
	}
	
	//epsilon greedy select action method
	public static action selectAction(State state, double epsilon) {
		//random action
		if(Math.random() < epsilon) {
			action[] actions = action.values();
			return actions[(int) (Math.random() * actions.length)];
		}else {
	        //greedy action
	        return qTable.get(state).entrySet().stream().max((entry1, entry2) -> Double.compare(entry1.getValue(), entry2.getValue())).get().getKey();
	    }
	}
	
	
	//this method outputs the policy in a printed form 
	private static void printPolicy(List<List<Character>> grid, boolean unicode) {
	    for (int i = 0; i < grid.size(); i++) {
	        for (int j = 0; j < grid.get(i).size(); j++) {
	            State state = new State(i, j);
	            char cell = grid.get(i).get(j);

	            if (cell == 'M' || cell == 'G' || cell == 'B') {
	                System.out.print(cell + " ");
	            } else if (qTable.containsKey(state)) {
	                action bestAction = selectGreedyAction(state);    
	                if (unicode) {
	                    System.out.print(getUnicodeSymbol(bestAction) + " ");
	                } else {
	                    System.out.print(getAsciiSymbol(bestAction) + " ");
	                }	                
	            } else {
	                System.out.print("? "); //unrecognized input 
	            }
	        }
	        System.out.println();
	    }
	}
	
	//this method evaluates the current policy 
	private static double evaluatePolicy(Map<State, Map<action, Double>> qTable, List<List<Character>> grid, int episodes, double successProbability) {
	    State startState = findStart(grid);
	    double totalReward = 0.0;
	    int maxSteps = grid.size() * grid.get(0).size(); //added a max steps limit to avoid infinite loops 

	    for (int i = 0; i < episodes; i++) {
	        State currentState = startState;
	        double episodeReward = 0.0;

	        for (int step = 0; step < maxSteps; step++) {
	            action bestAction = selectGreedyAction(currentState); //select the best option

	            State nextState = performAction(currentState, bestAction, successProbability, grid);
	            char cellType = grid.get(nextState.row).get(nextState.column);

	            episodeReward += getReward(cellType);

	            if (cellType == 'G' || cellType == 'M') { //terminal cells
	                break;
	            }

	            currentState = nextState;
	        }

	        totalReward += episodeReward;
	    }

	    return totalReward / episodes;
	}

	//this method selects the best option
	private static action selectGreedyAction(State state) {
	    return qTable.get(state).entrySet().stream()
	            .max((entry1, entry2) -> Double.compare(entry1.getValue(), entry2.getValue()))
	            .get().getKey();
	}
		
	//returns the ascii symbol 
	private static char getAsciiSymbol(action action) {
	    switch (action) {
	        case LEFT: return '<';
	        case UP: return '^';
	        case RIGHT: return '>';
	        case DOWN: return 'v';
	        default: return '?';
	    }
	}

	//returns the unicode symbol 
	private static String getUnicodeSymbol(action action) {
	    switch (action) {
	        case LEFT: return "\u2190"; // <
	        case UP: return "\u2191";   // ^
	        case RIGHT: return "\u2192"; // >
	        case DOWN: return "\u2193"; // v
	        default: return "?";
	    }
	}
	
	//this method prints the q-values
	private static void printQValues(List<List<Character>> grid) {
	    int cellWidth = 10;
	    int gridWidth = grid.get(0).size(); 
	    int totalWidth = gridWidth * cellWidth + gridWidth + 1;

	    //top border
	    System.out.println("-".repeat(totalWidth));

	    for (int i = 0; i < grid.size(); i++) {
	        for (int subRow = 0; subRow < 4; subRow++) { 
	            StringBuilder line = new StringBuilder("|");

	            for (int j = 0; j < grid.get(i).size(); j++) {
	                State state = new State(i, j);
	                char cell = grid.get(i).get(j);

	                //get qvalues if the state exists 
	                if (qTable.containsKey(state)) {
	                    Map<action, Double> qValues = qTable.get(state);

	                    double qUp = qValues.getOrDefault(action.UP, 0.0);
	                    double qDown = qValues.getOrDefault(action.DOWN, 0.0);
	                    double qLeft = qValues.getOrDefault(action.LEFT, 0.0);
	                    double qRight = qValues.getOrDefault(action.RIGHT, 0.0);

	                    //add qvalue 
	                    if (subRow == 0) {
	                        line.append(String.format(" %6.1f  |", qUp)); 
	                    } else if (subRow == 1) {
	                        line.append(String.format("%-6.1f   |", qLeft));
	                    } else if (subRow == 2) {
	                        line.append(String.format("   %6.1f|", qRight)); 
	                    } else if (subRow == 3) {
	                        line.append(String.format(" %6.1f  |", qDown));
	                    }
	                } else {
	                    //terminal cells 
	                    if (subRow == 0 || subRow == 3) {
	                        line.append("    0.0   |");
	                    } else {
	                        line.append("  0.0     |");
	                    }
	                }
	            }
	            System.out.println(line);
	        }
	        //seperate rows/bottom
	        System.out.println("-".repeat(totalWidth));
	    }
	}
}


//used a state class to represent the state of an agent
class State{
	int row;
	int column;
	
	public State(int row, int column) {
		this.row = row;
		this.column = column;
	}
	
	//this method checks and compares the parameters state, and its current state
	@Override
	public boolean equals(Object curr) {
		if (curr == null || getClass() != curr.getClass()) return false;
		State state = (State) curr;
		return (state.row == this.row) && (this.column == state.column);
	}
	
	//using a hashmap to store the q-values
	@Override
	public int hashCode() {
		return 31 * row + column; //31 is a prime number, so less collisions
	}
	
	public String outputState(int row, int column) {
		return "(" + row + "," + column + ")"; //returns states in a (x, y) format
	}
}
