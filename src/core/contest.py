def solve_game_show(R, P):
    MOD = 10**9 + 7
    
    # dp[round][bebo_wins] = number of ways to reach this state
    dp = [[0] * (R + 1) for _ in range(R + 1)]
    dp[0][0] = 1  # Initially, no rounds played, no wins for Bebo
    
    for round_num in range(1, R + 1):
        for bebo_wins in range(round_num + 1):
            lolo_wins = round_num - bebo_wins
            
            # Check if this state satisfies the constraint
            # Bebo needs: bebo_wins >= P * lolo_wins
            if bebo_wins >= P * lolo_wins:
                # This round Bebo wins
                if bebo_wins > 0:
                    dp[round_num][bebo_wins] = (dp[round_num][bebo_wins] + 
                                               dp[round_num - 1][bebo_wins - 1]) % MOD
                
                # This round Lolo wins
                if lolo_wins > 0:
                    dp[round_num][bebo_wins] = (dp[round_num][bebo_wins] + 
                                               dp[round_num - 1][bebo_wins]) % MOD
    
    # Count total ways where Bebo wins the show
    # Bebo needs more than half the rounds
    min_wins_to_win_show = R // 2 + 1
    
    result = 0
    for bebo_wins in range(min_wins_to_win_show, R + 1):
        result = (result + dp[R][bebo_wins]) % MOD
    
    return result

def main():
    T = int(input())
    for _ in range(T):
        R, P = map(int, input().split())
        print(solve_game_show(R, P))

if _name_ == "_main_":
    main()