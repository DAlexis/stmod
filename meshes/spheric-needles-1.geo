// Gmsh project created on Sat Sep 28 14:07:03 2019
SetFactory("OpenCASCADE");

// For size in m
scale = 1.0e-3;

// For size in mm
//scale = 1.0;

needle_rad = 200e-3 * scale;
//needle_len = 5 * scale;
//cyl_height = 20 * scale;

needle_len = 2 * scale;
cyl_height = 17 * scale;

cyl_rad = 6 * scale;

//+

// Bottom plane
Point(1) = {0.0, 0, 0, needle_rad};


// Top needle
Point(6) = {0, cyl_height-needle_len-needle_rad, 0, needle_rad / 12.0};
Point(7) = {0, cyl_height-needle_len, 0, needle_rad / 12.0};
Point(8) = {needle_rad, cyl_height-needle_len, 0, needle_rad / 12.0};
//Point(9) = {needle_rad, cyl_height, 0, needle_rad / 2.0};
Point(9) = {needle_rad, cyl_height, 0, cyl_rad / 7.0};

// Contour
Point(10) = {cyl_rad, cyl_height, 0, cyl_rad / 7.0};
Point(11) = {cyl_rad, 0, 0, cyl_rad / 7.0};

Line(1) = {8, 9};
Line(2) = {9, 10};
Line(3) = {10, 11};
Line(4) = {11, 1};
Line(5) = {1, 6};

Circle(6) = {6, 7, 8};

Curve Loop(1) = {5, 6, 1, 2, 3, 4};
Plane Surface(1) = {1};

Physical Surface(2) = {1};

// some parameters for the meshing:
Mesh.Algorithm = 1;
Mesh.RecombineAll = 1;
Mesh.CharacteristicLengthFactor = 1.0;
Mesh.SubdivisionAlgorithm = 1;
Mesh.Smoothing = 200;
Show "*";

//Recombine Surface {1};
