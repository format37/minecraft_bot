import sys
import subprocess
from pathlib import Path
from typing import List
import math
import numpy as np
import pyvista as pv
from pydantic import BaseModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from PIL import Image
import io
from request_to_scad import generate_scad_script, encode_image
import logging
from datetime import datetime as dt

logging.basicConfig(level=logging.INFO)

class Row(BaseModel):
    blocks: List[int]

class Layer(BaseModel):
    rows: List[Row]

class Object3D(BaseModel):
    layers: List[Layer]
    material: str
    description: str

def numpy_to_object3d(voxel_data: np.ndarray, material: str = "default") -> Object3D:
    layers = []
    for z in range(voxel_data.shape[2]):
        rows = []
        for y in range(voxel_data.shape[1]):
            blocks = voxel_data[:, y, z].tolist()
            rows.append(Row(blocks=blocks))
        layers.append(Layer(rows=rows))
    
    return Object3D(layers=layers, material=material, description="")

def print_object3d(obj: Object3D):
    # print(f"Material: {obj.material}")
    # print(f"Object dimensions: {len(obj.layers[0].rows[0].blocks)} x {len(obj.layers[0].rows)} x {len(obj.layers)}")
    # print()
    logging.info(f"Material: {obj.material}")
    logging.info(f"Object dimensions: {len(obj.layers[0].rows[0].blocks)} x {len(obj.layers[0].rows)} x {len(obj.layers)}")
    logging.info("")

    for layer_index, layer in enumerate(obj.layers):
        # print(f"Layer {layer_index + 1}:")
        logging.info(f"Layer {layer_index + 1}:")
        for row in layer.rows:
            # print(''.join(['█' if block else '·' for block in row.blocks]))
            logging.info(''.join(['█' if block else '·' for block in row.blocks]))
        # print()  # Empty line between layers
        logging.info("")  # Empty line between layers

def openscad_to_stl(scad_file: Path, stl_file: Path):
    cmd = ['openscad', '-o', str(stl_file), str(scad_file)]
    subprocess.run(cmd, check=True)

def render_voxels(voxels, mesh):
    # Create a plotter object
    p = pv.Plotter()

    # Add the voxel grid to the plotter
    p.add_mesh(voxels, color=True, show_edges=True, opacity=0.5)

    # Optionally, add the original mesh for comparison
    p.add_mesh(mesh, color="lightblue", opacity=0.5)

    # Show the plot
    p.show()

def custom_downscale_3d(arr, scale_factor):
    # Ensure the array dimensions are divisible by scale_factor
    pad_width = [(0, (scale_factor - s % scale_factor) % scale_factor) for s in arr.shape]
    arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
    
    # Reshape the array into scale_factor x scale_factor x scale_factor cubes
    shape = (arr.shape[0]//scale_factor, scale_factor, 
             arr.shape[1]//scale_factor, scale_factor, 
             arr.shape[2]//scale_factor, scale_factor)
    reshaped = arr.reshape(shape)
    
    # Apply the custom logic: if any element in a cube is 1, the result is 1
    return np.max(reshaped, axis=(1,3,5))

def openscad_to_voxels(scad_file: Path, target_size: int = 10) -> Object3D:
    # Convert OpenSCAD to STL
    stl_file = scad_file.with_suffix('.stl')
    openscad_to_stl(scad_file, stl_file)

    # prefix = dt.now().strftime("%Y%m%d_%H%M%S_")
    # copy_of_stl_file = Path(f'./objects/{prefix}model.stl')
    # openscad_to_stl(scad_file, copy_of_stl_file)
    
    # Read the STL file
    mesh = pv.read(stl_file)
    
    # Set the density of the voxel grid
    density = mesh.length / 100
    # print(f"Density: {density}")
    logging.info(f"Density: {density}")

    # ++
    # Voxelize the mesh
    voxels = pv.voxelize(mesh, density=density, check_surface=False)

    # Compute the implicit distance of the voxels
    voxels.compute_implicit_distance(mesh, inplace=True)

    # Create a structured grid from the voxel data
    x_min, x_max, y_min, y_max, z_min, z_max = voxels.bounds
    x = np.arange(x_min, x_max, density)
    y = np.arange(y_min, y_max, density)
    z = np.arange(z_min, z_max, density)
    grid = pv.StructuredGrid(*np.meshgrid(x, y, z, indexing='ij'))

    # Select enclosed points to create a mask
    selection = grid.select_enclosed_points(mesh.extract_surface(), tolerance=0.0, check_surface=False)
    mask = selection['SelectedPoints'].view(np.bool_)

    # Reshape the mask to match the voxel grid dimensions
    mask_reshaped_original = mask.reshape(grid.dimensions, order='F')  # Use 'F' for Fortran-style order 

    # print(f"mask_reshaped_original shape: {mask_reshaped_original.shape}")  
    logging.info(f"mask_reshaped_original shape: {mask_reshaped_original.shape}")

    current_shape = mask_reshaped_original.shape

    # # Downscale with custom logic
    downscale_factor = 2
    if max(current_shape) < target_size:
        # print(f"Don't need to downscale")
        logging.info(f"Don't need to downscale")
        mask_reshaped = mask_reshaped_original
    else:
        while max(current_shape) > target_size:
            mask_reshaped = custom_downscale_3d(mask_reshaped_original, downscale_factor)
            # print(f"mask_reshaped shape ({max(mask_reshaped.shape)}): {mask_reshaped.shape}")
            logging.info(f"mask_reshaped shape ({max(mask_reshaped.shape)}): {mask_reshaped.shape}")
            current_shape = mask_reshaped.shape
            downscale_factor += 1

    # Now, mask_reshaped is your NumPy array representing the voxel data
    # Save the voxel data to a file
    np.save('voxel_data.npy', mask_reshaped)
    # print(f"Voxel data ones length: {np.sum(mask_reshaped)}")
    logging.info(f"Voxel data ones length: {np.sum(mask_reshaped)}")
    # Print shape
    # print(f"Voxel data shape: {mask_reshaped.shape}")
    logging.info(f"Voxel data shape: {mask_reshaped.shape}")
    return voxels, mesh

def plot_voxels_matplotlib_single(voxel_data):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the voxels with opacity 0.5
    ax.voxels(voxel_data, edgecolor='k', facecolors='#1f77b4', alpha=0.4)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Show the plot
    plt.show()

def plot_voxels_matplotlib(voxel_data, elev, azim):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the voxels with opacity 0.5
    ax.voxels(voxel_data, edgecolor='k', facecolors='#1f77b4', alpha=0.8)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set the viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # Instead of showing the plot, save it to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Close the plot to free up memory
    plt.close(fig)
    
    return Image.open(buf)

def create_combined_image(voxel_data):
    # Create four views
    views = [
        plot_voxels_matplotlib(voxel_data, 30, 45),   # Front-right
        plot_voxels_matplotlib(voxel_data, 30, 225),  # Back-left
        plot_voxels_matplotlib(voxel_data, 30, 135),  # Front-left
        plot_voxels_matplotlib(voxel_data, 30, 315)   # Back-right
    ]
    
    # Create a new image with 2x2 grid
    combined_image = Image.new('RGB', (800, 800))
    
    # Paste the four views into the grid
    combined_image.paste(views[0], (0, 0))
    combined_image.paste(views[1], (400, 0))
    combined_image.paste(views[2], (0, 400))
    combined_image.paste(views[3], (400, 400))
    
    return combined_image

def plot_voxels_plotly(voxel_data):
    # Get the coordinates of the voxels
    x, y, z = np.where(voxel_data)
    
    # Create the 3D scatter plot for voxels
    voxels = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
            opacity=0.1
        ),
        name='Voxels'
    )
    
    # Create the edges
    edges_x, edges_y, edges_z = [], [], []
    shape = voxel_data.shape
    for i in range(len(x)):
        for dx, dy, dz in [(1,0,0), (0,1,0), (0,0,1)]:
            nx, ny, nz = x[i]+dx, y[i]+dy, z[i]+dz
            if nx < shape[0] and ny < shape[1] and nz < shape[2] and voxel_data[nx, ny, nz]:
                edges_x.extend([x[i], nx, None])
                edges_y.extend([y[i], ny, None])
                edges_z.extend([z[i], nz, None])
    
    edges = go.Scatter3d(
        x=edges_x, y=edges_y, z=edges_z,
        mode='lines',
        line=dict(color='black', width=1),
        name='Edges'
    )
    
    # Combine voxels and edges
    fig = go.Figure(data=[voxels, edges])
    
    # Update the layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=800,
        title='3D Voxel Plot with Edges'
    )
    
    # Show the plot
    fig.show()

def generate_blueprint_from_description(user_request, size_x, size_y, size_z, steps_limit):
    need_to_continue = True

    # scad_script = generate_scad_script()
    # size = 20
    # user_request = "Modern house"
    step = 0
    # steps_limit = 7
    request = '{task: "'+user_request+'". Size in blocks (x,y,z): ('+str(size_x)+','+str(size_y)+','+str(size_z)+'), step: '+str(step)+', scad_script:"", next_step_plan: ""}'
    base64_image = None
    object_3d = None

    while need_to_continue:
        # print(f"Step: {step}")
        logging.info(f"Step: {step}")
        task, scad, plan, need_to_continue, design_description = generate_scad_script(request, base64_image, steps_limit)
        # print(f"Scad script: {scad}")
        # Save the script to a file
        scad_file = Path('model.scad')
        with open(scad_file, 'w') as f:
            f.write(scad)

        prefix = dt.now().strftime("%Y%m%d_%H%M%S_")
        copy_of_scad_file = Path(f'objects/{prefix}model_{step}.scad')
        with open(copy_of_scad_file, 'w') as f:
            f.write(scad)

        # scad_path = Path(scad_file)
        # if not scad_path.exists():
        #     print(f"Error: File {scad_file} does not exist.")
        #     sys.exit(1)
        
        # Convert OpenSCAD to voxels
        try:
            size = max(size_x, size_y, size_z)
            voxels, mesh = openscad_to_voxels(scad_file,size)
        except Exception as e:
            # print(f"Error: {e}")
            logging.error(f"Error: {e}")
            # print("Trying to repeat the step")
            logging.info("Trying to repeat the step")
            continue
        
        # Load the saved voxel data
        voxel_data = np.load('voxel_data.npy')
        
        # Convert NumPy array to Object3D
        object_3d = numpy_to_object3d(voxel_data)
        object_3d.description = design_description
        # Print the Object3D representation
        # print_object3d(object_3d)

        # Render the voxels
        # render_voxels(voxels, mesh)
        # plot_voxels_matplotlib(voxel_data)
        # Create and save the combined image
        result = create_combined_image(voxel_data)
        result.save('combined.png')
        prefix = dt.now().strftime("%Y%m%d_%H%M%S_")
        result.save(f'objects/{prefix}combined_{step}.png')
        # plot_voxels_plotly(voxel_data)

        base64_image = encode_image('combined.png')
        step += 1
        if step > steps_limit:
            break
        request = '{task: "'+task+'", step: '+str(step)+', scad_script:"'+scad+'", next_step_plan: "'+plan+'"}'

        
    # print("Done")
    logging.info("generate_blueprint_from_description - Done")
    return object_3d


def main():
    
    obj = generate_blueprint_from_description(
        user_request = "Modern house", 
        size = 20,
        steps_limit = 3
    )
    print_object3d(obj)

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python script.py <path_to_openscad_file>")
    #     sys.exit(1)
    
    # main(sys.argv[1])
    main()