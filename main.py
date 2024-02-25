import os
os.environ['MAX_JOBS'] = '4'
   
def run(args):
    if args.julia:
        from julia import visualize_julia_set
        visualize_julia_set(scale=args.scale)
    if args.mandelbrot:
        from mandelbrot import visualize_mandelbrot_set
        visualize_mandelbrot_set(scale=args.scale)
        
    if not args.julia and not args.mandelbrot:
        from mandelbrot import visualize_mandelbrot_set
        from julia import visualize_julia_set
        
        visualize_julia_set(scale=args.scale)
        visualize_mandelbrot_set(scale=args.scale)
    
    return 0
 
if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Visualize Julia and Mandelbrot sets')
    
    parser.add_argument('--julia', action='store_true', help='Visualize Julia set')
    parser.add_argument('--mandelbrot', action='store_true', help='Visualize Mandelbrot set')
    parser.add_argument('--scale', type=int, default=1, help='fine-scale factor')
    
    args = parser.parse_args()
    
    sys.exit(run(args))    
