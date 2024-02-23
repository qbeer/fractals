import os
os.environ['MAX_JOBS'] = '4'
   
def run(args):
    if args.julia:
        from julia import visualize_julia_set
        visualize_julia_set()
    if args.mandelbrot:
        from mandelbrot import visualize_mandelbrot_set
        visualize_mandelbrot_set()
        
    if not args.julia and not args.mandelbrot:
        print('Please specify --julia and/or --mandelbrot')
    
    return 0   
 
if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Visualize Julia and Mandelbrot sets')
    
    parser.add_argument('--julia', action='store_true', help='Visualize Julia set')
    parser.add_argument('--mandelbrot', action='store_true', help='Visualize Mandelbrot set')
    
    args = parser.parse_args()
    
    sys.exit(run(args))    
