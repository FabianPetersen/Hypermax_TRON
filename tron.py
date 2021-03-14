import numpy as np
import sys
import operator
from heapq import *
import concurrent.futures

K1 = 55
K2 = 194
K3 = 3


class Game:
    def __init__(self, width=30, height=20):
        """
        Init the game

        :param width:
        :param height:
        """
        # Width / height of the board
        self.width = width
        self.height = height

        # Game board
        self.current_state = np.array([['.' for y in range(0, height)] for x in range(0, width)])

        # Articulation points
        self.articulation_points = np.zeros((self.width, self.height))

        # Map of neighbors
        self.game_board = {}
        self.generate_neighbors()

        # Player X always plays first
        self.player_turn = 'X'

    def generate_neighbors(self):
        for x in range(0, self.width):
            for y in range(0, self.height):
                self.game_board[(x, y)] = self.get_neighbors(x, y)

    def draw_board(self):
        """
        Draw the game board

        :return:
        """
        for x in range(0, self.width):
            for y in range(0, self.height):
                print('{} |'.format(self.current_state[x, y]), end=" ")
            print()
        print()

    def get_neighbors(self, x=0, y=0):
        """
        Get all valid neighbors for a point

        :param x:
        :param y:
        :return:
        """
        # Possible moves
        moves = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]

        # Valid moves
        possible = []
        for move in moves:
            # If move is within bounds and empty
            if 0 <= move[0] < self.width and 0 <= move[1] < self.height and self.current_state[move[0], move[1]] == '.':
                possible.append(move)

        return possible

    def remove_from_neighbors(self, x, y):
        """
        Make the neighbors not traversable
        :param x:
        :param y:
        :return:
        """
        neighbors = self.get_fast_neighbors(x, y)

        for n_x, n_y in neighbors:
            self.game_board[n_x, n_y].remove((x, y))


    def get_fast_neighbors(self, x, y, should_filter=False):
        return self.game_board[(x, y)] if not should_filter else list(filter(lambda move: self.current_state[move[0], move[1]] == '.', self.game_board[(x, y)]))

    def is_end(self, x=0, y=0):
        """
        Is this an end state for a player

        :param x:
        :param y:
        :return:
        """
        return len(self.get_neighbors(x, y)) == 0

    def hypermax(self, heads=[(0, 0)], current_head_index=0, alpha=[-999999], depth=3):
        """
        https://www.diva-portal.org/smash/get/diva2:761634/FULLTEXT01.pdf

        :return:
        """
        max_m = []
        best_move = (0, 0)

        head = heads[current_head_index]

        # If the game came to an end, the function needs to return
        # the evaluation function of the end.
        if self.is_end(head[0], head[1]):
            m = [1 for x in range(len(heads))]
            m[current_head_index] = 0
            return Game.get_hypermax_return(m), (0, 0)

        elif depth == 0:
            return Game.get_hypermax_return(self.voronoi_heuristic_hypermax(heads)), (0, 0)

        first_child = True
        for move in self.get_fast_neighbors(head[0], head[1]):
            # Check if the move is valid
            if self.current_state[move[0], move[1]] != '.':
                continue

            # Make the move
            self.current_state[move[0], move[1]] = Game.get_char(current_head_index)

            # Move the head
            new_heads = heads[:]
            new_heads[current_head_index] = move

            # Get next player move
            m, _ = self.hypermax(new_heads, (current_head_index + 1) % len(heads), alpha[:], depth - 1)

            # Setting back the field to empty
            self.current_state[move[0], move[1]] = '.'

            if first_child:
                max_m = m
                best_move = move

            if alpha[current_head_index] < m[current_head_index]:
                alpha[current_head_index] = m[current_head_index]
                max_m = m
                best_move = move

            if sum(alpha) >= 0:
                break

            first_child = False

        return max_m, best_move

    def dijkstra(self, head):
        dists = np.zeros((self.width, self.height))
        dists[:] = np.inf
        dists[head[0], head[1]] = 0.0
        visited_ap = set()

        q, seen = [(0, head)], set()
        while q:
            (cost, v1) = heappop(q)
            if v1 not in seen:
                dists[v1[0], v1[1]] = cost
                seen.add(v1)

                for v2 in self.get_fast_neighbors(v1[0], v1[1]):
                    # Check if the move is valid, and not a articulation point
                    if v2 in seen or self.current_state[v2[0], v2[1]] != '.':
                        if self.articulation_points[v2[0], v2[1]] > 0 and v2 not in visited_ap:
                            visited_ap.add(v2)
                        continue

                    heappush(q, (cost + 1, v2))

        return dists, visited_ap

    def voronoi_heuristic_hypermax(self, heads):
        count = [0 for x in range(len(heads))]
        edges = [0 for x in range(len(heads))]
        maxcost = 1000000

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        # Set articulation points
        self.articulation_points = self.find_articulation_points(heads)

        # Use dijkstra to calculate cost
        # costs_future = [executor.submit(self.dijkstra, heads[i]) for i in range(len(heads))]
        # costs = [future.result() for future in costs_future]
        costs = [self.dijkstra(head) for head in heads]

        for x in range(0, self.width):
            for y in range(0, self.height):
                point_costs = [costs[i][0][x, y] for i in range(len(costs))]
                min_index, min_value = min(enumerate(point_costs), key=operator.itemgetter(1))

                # This bot has the best value for that square
                if min_value <= maxcost and point_costs.count(min_value) == 1:
                    # Count open edges and nodes
                    count[min_index] += 1
                    edges[min_index] += len(self.get_fast_neighbors(x, y, True))

        return [(K1 * count[i] + K2 * edges[i]) / maxcost for i in range(len(heads))]

    def _find_articulation_points_util(self, u, visited, ap, parent, low, disc, depth=0):
        # Count of children in current node
        children = 0

        # Mark the current node as visited
        visited[u[0], u[1]] = 1

        # Initialize discovery time and low value
        disc[u[0], u[1]] = depth
        low[u[0], u[1]] = depth

        # Recur for all the vertices adjacent to this vertex
        for v in self.get_fast_neighbors(u[0], u[1]):
            if self.current_state[v[0], v[1]] != '.':
                continue

            # If v is not visited yet, then make it a child of u
            # in DFS tree and recur for it
            if visited[v[0], v[1]] == 0:
                parent[v[0], v[1]] = Game.get_point_id(u[0], u[1])
                children += 1
                self._find_articulation_points_util(v, visited, ap, parent, low, disc, depth+1)

                # Check if the subtree rooted with v has a connection to
                # one of the ancestors of u
                low[u] = min(low[u[0], u[1]], low[v[0], v[1]])

                # u is an articulation point in following cases
                # (1) u is root of DFS tree and has two or more children.
                if parent[u[0], u[1]] == -1 and children > 1:
                    ap[u[0], u[1]] = 1

                # (2) If u is not root and low value of one of its child is more
                # than discovery value of u.
                if parent[u[0], u[1]] != -1 and low[v[0], v[1]] >= disc[u[0], u[1]]:
                    ap[u[0], u[1]] = 1

            # Update low value of u for parent function calls
            elif Game.get_point_id(v[0], v[1]) != parent[u[0], u[1]]:
                low[u] = min(low[u[0], u[1]], disc[v[0], v[1]])

    # The function to do DFS traversal. It uses recursive APUtil()
    def find_articulation_points(self, starting_points=[(0, 0)]):
        """
            A recursive function that find articulation points
            using DFS traversal

            u --> The vertex to be visited next
            visited[] --> keeps tract of visited vertices
            disc[] --> Stores discovery times of visited vertices
            parent[] --> Stores parent vertices in DFS tree
            ap[] --> Store articulation points
        """

        # Mark all the vertices as not visited
        # and Initialize parent and visited,
        # and ap(articulation point) arrays
        visited = np.zeros((self.width, self.height))
        low = np.zeros((self.width, self.height))
        low[:] = np.inf
        disc = np.zeros((self.width, self.height))
        disc[:] = np.inf
        ap = np.zeros((self.width, self.height))
        parent = np.zeros((self.width, self.height))
        parent[:] = np.inf
        for point in starting_points:
            parent[point[0], point[1]] = -1

        # Call the recursive helper function
        # to find articulation points
        # in DFS tree rooted with vertex 'i'
        for (x, y) in starting_points:
            for point in self.get_fast_neighbors(x, y):
                if self.current_state[point[0], point[1]] == '.' and visited[point[0], point[1]] == 0:
                    self._find_articulation_points_util(point, visited, ap, parent, low, disc, 1)

        return ap

    def get_direction(self, origin, destination):
        """

        :return:
        """
        if origin[0] == destination[0]:
            if origin[1] > destination[1]:
                return "UP"
            else:
                return "DOWN"

        else:
            if origin[0] > destination[0]:
                return "LEFT"
            else:
                return "RIGHT"

    def delete_player(self, index):
        char = Game.get_char(index)
        for x in range(0, self.width):
            for y in range(0, self.height):
                if char == self.current_state[x, y]:
                    self.current_state[x, y] = '.'

    def play(self):
        # Set positions
        heads = [(0, 0), (int(self.width / 2), int(self.height / 2)), (self.width-1, self.height-1)]
        depth = len(heads) - 1

        # Set head
        self.current_state[heads[0][0], heads[0][1]] = Game.get_char(0)
        self.current_state[heads[1][0], heads[1][1]] = Game.get_char(1)
        self.current_state[heads[2][0], heads[2][1]] = Game.get_char(2)

        i = 100
        while i > 0:
            i -= 1

            # If it's player's turn
            if self.player_turn == Game.get_char(0):
                self.draw_board()

                # (m, qx, qy) = self.min_alpha_beta(-2, 2, heads, 0, depth=3)
                m, (qx, qy) = self.hypermax(heads, 0, [-999999 for x in range(len(heads))], depth=depth)
                print('Recommended move: X = {}, Y = {}'.format(qx, qy))

                # px = int(input('Insert the X coordinate: '))
                # py = int(input('Insert the Y coordinate: '))
                # input('wait ')
                (px, py) = qx, qy

                heads[0] = (px, py)
                self.remove_from_neighbors(px, py)
                self.current_state[px, py] = Game.get_char(0)
                self.player_turn = Game.get_char(1)

            # If it's AI's turn
            elif self.player_turn == Game.get_char(1):
                m, (px, py) = self.hypermax(heads, 1, [-999999 for x in range(len(heads))], depth=depth)

                print(m, px, py)
                heads[1] = (px, py)
                self.remove_from_neighbors(px, py)
                self.current_state[px, py] = Game.get_char(1)
                self.player_turn = Game.get_char(2)

            else:
                m, (px, py) = self.hypermax(heads, 2, [-999999 for x in range(len(heads))], depth=depth)

                print(m, px, py)
                heads[2] = (px, py)
                self.remove_from_neighbors(px, py)
                self.current_state[px, py] = Game.get_char(2)
                self.player_turn = Game.get_char(0)

    def play_codingame(self):
        heads = []
        heads_indexes = {}
        dead_players = []

        # game loop
        while True:
            # n: total number of players (2 to 4).
            # p: your player number (0 to 3).
            n, p = [int(i) for i in input().split()]
            for i in range(n):
                # x0: starting X coordinate of lightcycle (or -1)
                # y0: starting Y coordinate of lightcycle (or -1)
                # x1: starting X coordinate of lightcycle (can be the same as X0 if you play before this player)
                # y1: starting Y coordinate of lightcycle (can be the same as Y0 if you play before this player)
                x0, y0, x1, y1 = [int(j) for j in input().split()]
                print('Player{}: X={} Y={}'.format(i, x1, y1), file=sys.stderr, flush=True)

                # The player is dead
                if i in dead_players:
                    continue

                if x1 > -1 and y1 > -1 and len(heads) != n:
                    heads_indexes[i] = i
                    heads.append((x1, y1))

                # Remove the dead player
                if x1 == -1 and y1 == -1:
                    self.delete_player(i)
                    del heads[i]

                    # Change the indexes
                    for a in range(i, n):
                        heads_indexes[a] -= 1

                    dead_players.append(i)

                    continue

                # Set positions
                heads[heads_indexes[i]] = (x1, y1)
                self.remove_from_neighbors(x1, y1)

                # Set head
                self.current_state[x1, y1] = Game.get_char(i)
                self.current_state[x0, y0] = Game.get_char(i)

            # Write an action using print
            # To debug: print("Debug messages...", file=sys.stderr, flush=True)

            # A single line with UP, DOWN, LEFT or RIGHT
            # (m, qx, qy) = self.min_alpha_beta(-1, 1, heads, heads_indexes[p], depth=1)
            (m, (qx, qy)) = self.hypermax(heads, heads_indexes[p], [-999999 for x in range(len(heads))], depth=n-1)
            print('Recommended move: X = {} Y = {}'.format(qx, qy), file=sys.stderr, flush=True)
            print('Current Value: {}'.format(self.current_state[qx, qy]), file=sys.stderr, flush=True)
            print(self.get_direction(heads[p], (qx, qy)))

    @staticmethod
    def get_point_id(x=0, y=0):
        return x * 1000 + y

    @staticmethod
    def get_char(index=0):
        return ['X', 'O', 'E', 'W'][index]

    @staticmethod
    def get_hypermax_return(values):
        avg = sum(values) / len(values)
        return [x - avg for x in values]


if __name__ == "__main__":
    g = Game()
    g.play()
    # g.play_codingame()
