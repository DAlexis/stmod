// Gmsh project created on Sat Sep 28 14:07:03 2019
SetFactory("OpenCASCADE");

needle_rad = 0.1;
needle_len = 1;
cyl_height = 4;
cyl_rad = 2;
//+

// Bottom needle
Point(1) = {needle_rad, 0, 0, 1.0};
Point(2) = {needle_rad, needle_len, 0, 1.0};
Point(4) = {0, needle_len + needle_rad, 0, 1.0};
Point(5) = {0, needle_len, 0, 1.0};


// Top needle
Point(6) = {0, cyl_height-needle_len-needle_rad, 0, 1.0};
Point(7) = {0, cyl_height-needle_len, 0, 1.0};
Point(8) = {needle_rad, cyl_height-needle_len, 0, 1.0};
Point(9) = {needle_rad, cyl_height, 0, 1.0};

// Contour
Point(10) = {cyl_rad, cyl_height, 0, 1.0};
Point(11) = {cyl_rad, 0, 0, 1.0};

//+
Line(1) = {2, 1};
//+
Line(2) = {1, 11};
//+
Line(3) = {11, 10};
//+
Line(4) = {10, 9};
//+
Line(5) = {9, 8};
//+
Line(6) = {6, 4};
//+
Circle(7) = {2, 5, 4};
//+
Circle(8) = {6, 7, 8};
//+
Curve Loop(1) = {5, -8, 6, -7, 1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
//Recombine Surface {1};
//+
//Recombine Surface {1};
//+
Recombine Surface {1};
