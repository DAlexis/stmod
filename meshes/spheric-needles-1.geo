// Gmsh project created on Sat Sep 28 14:07:03 2019
SetFactory("OpenCASCADE");

// Size in mm

needle_rad = 200e-3;
needle_len = 5;
cyl_height = 25;
cyl_rad = 15;
//+

// Bottom needle
Point(1) = {0.0, 0, 0, 1.0};
/*
Point(1) = {needle_rad, 0, 0, 1.0};
Point(2) = {needle_rad, needle_len, 0, 1.0};
Point(4) = {0, needle_len + needle_rad, 0, 1.0};
Point(5) = {0, needle_len, 0, 1.0};*/


// Top needle
Point(6) = {0, cyl_height-needle_len-needle_rad, 0, 1.0};
Point(7) = {0, cyl_height-needle_len, 0, 1.0};
Point(8) = {needle_rad, cyl_height-needle_len, 0, 1.0};
Point(9) = {needle_rad, cyl_height, 0, 1.0};

// Contour
Point(10) = {cyl_rad, cyl_height, 0, 1.0};
Point(11) = {cyl_rad, 0, 0, 1.0};

//+
Line(1) = {8, 9};
//+
Line(2) = {9, 10};
//+
Line(3) = {10, 11};
//+
Line(4) = {11, 1};
//+
Line(5) = {1, 6};
//+
Circle(6) = {6, 7, 8};
//+
Curve Loop(1) = {5, 6, 1, 2, 3, 4};
//+
Plane Surface(1) = {1};
