# Anthill-Calculator

This is my solution to the anthill puzzle listed on many of Optiver's job openings. The puzzle is stated as follows:

An ant leaves its anthill in order to forage for food. It moves with the speed of 10cm per second, but it doesn't know where to go, therefore every second it moves randomly 10cm directly north, south, east or west with equal probability.
1. If the food is located on east-west lines 20cm to the north and 20cm to the south, as well as on north-south lines 20cm to the east and 20cm to the west from the anthill, how long will it take the ant to reach it on average?
2. What is the average time the ant will reach food if it is located only on a diagonal line passing through (10cm, 0cm) and (0cm, 10cm) points?
3. Can you write a program that comes up with an estimate of average time to find food for any closed boundary around the anthill? What would be the answer if food is located outside an defined by ( (x – 2.5cm) / 30cm )2 + ( (y – 2.5cm) / 40cm )2 < 1 in coordinate system where the anthill is located at (x = 0cm, y = 0cm)? Provide us with a solution rounded to the nearest integer.

The program anthill_calculator.py answers part 3 of the question. For my full solution (and to understand what the program is actually doing), see the file anthill_puzzle_solution.pdf.

My solution is based on the techniques found in Mikey Wright's master's thesis, 'Boundary Problems for One and Two Dimensional Random Walks'; in particular, chapter 4. Essentially, we assign a variable to each point on the interior representing the average number of steps required to reach the boundary starting from that point. The problem can then be fully encoded as a system of linear equations, which can be efficiently solved to give the average time required to hit the boundary from any starting position.
