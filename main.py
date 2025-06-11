from utils.visualization_utils import display_paths
import plotly.io as pio
from datetime import datetime
import gc

""" Note to Self """
# example use case 1
# work on making the equation user-inputted. right now, it's a set equation.

def main():
    gc.enable()
    try:
        initial_guess = -0.5
        step_size = 0.05
        iterations = 20
        print("Displaying 2D & 3D Gradient Descent Paths...")
        fig = display_paths(initial_guess, 1.0, step_size, iterations)
        pio.write_html(
            fig, 
            file="output.html", 
            auto_open=False
        )
    except Exception as e: raise (e)
    finally: 
        print(f"============ Logged Success at {datetime.now().strftime('%Y-%m-%d')} ============\n")
        del fig, step_size, initial_guess

# main()
if __name__ == "__main__":
    main()