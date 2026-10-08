import open3d as o3d
import time

input_file = "generic-car-frame-chasis-and-body-in-white-1.snapshot.1/car frame biw chassis.stl"
output_file = "generic-car-frame-chasis-and-body-in-white-1.snapshot.1/car_simplified.stl"

print(f"Loading mesh: {input_file}")
mesh = o3d.io.read_triangle_mesh(input_file)
print(f"Original faces: {len(mesh.triangles)}")

print("Simplifying mesh to 150,000 faces...")
t0 = time.time()
mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=150000)
print(f"Simplified faces: {len(mesh_smp.triangles)} in {time.time()-t0:.2f}s")

print(f"Saving to {output_file}")
o3d.io.write_triangle_mesh(output_file, mesh_smp)
print("Done!")
