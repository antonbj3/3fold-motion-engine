| index | scene | n_b | n_c | max_degree | mass_ratio | mu_min | mu_max | lambda_min_pos | lambda_max | condition | origin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | cube | 1 | 4 | 4 | 1.0 | 0.5 | 0.5 | 0.11169631197754926 | 0.9549703546891176 | 8.54970354689119 | rigid box pushed on a plane; four corner point contacts |
| 1 | eight_box_tower | 8 | 32 | 8 | 1.0 | 0.5 | 0.5 | 0.0023331472395808376 | 46.37933350570455 | 19878.44261129308 | eight-box tower; point contacts between consecutive boxes and the ground |
| 2 | three_body_column_normal | 3 | 3 | 2 | 1000.0 | 0.5 | 0.5 | 0.0002927682519584372 | 2.0010002499999997 | 6834.758333987907 | three-body point-mass column with mass ratio 1000:1, normal approach |
| 3 | three_body_column_tangential_drive | 3 | 3 | 2 | 1000.0 | 0.5 | 0.5 | 0.0002927682519584372 | 2.0010002499999997 | 6834.758333987907 | three-body point-mass column with mass ratio 1000:1, normal approach with tangential drive |
| 4 | three_body_column_separating | 3 | 3 | 2 | 1000.0 | 0.5 | 0.5 | 0.0002927682519584372 | 2.0010002499999997 | 6834.758333987907 | three-body point-mass column with mass ratio 1000:1, separating initial velocities |
| 5 | eight_box_tower_mass_imbalance | 8 | 32 | 8 | 1000.0 | 0.5 | 0.5 | 3.778713139723853e-06 | 24.021194460346837 | 6356977.513805744 | eight-box tower with alternating mass ratio 1000:1 and a free-velocity perturbation |
| 6 | lattice | 8 | 64 | 16 | 1.0 | 0.5 | 0.5 | 0.40211196847168934 | 65.1599471372697 | 162.04428678142486 | 2x2x2 box lattice with patch contacts |
| 7 | random_contact_graph_256 | 64 | 256 | 13 | 1.0 | 0.5 | 0.5 | 0.21634056493239456 | 25.761485549826027 | 119.07838716181763 | synthetic random contact graph, 64 bodies and 256 contacts, fixed seed |
| 8 | granular_press_step | 125 | 441 | 14 | 1.0 | 0.4 | 0.5 | 0.6172496688862671 | 526.1260007131912 | 852.3714588012746 | granular DEM press step, 125 grains and 441 contacts |
| 9 | sphere_packing_50 | 50 | 176 | 10 | 1.0 | 0.5 | 0.5 | 4.93522895448179 | 4306.254112147977 | 872.5540703106333 | random sphere packing, 50 grains |
| 10 | sphere_packing_200 | 200 | 710 | 10 | 1.0 | 0.5 | 0.5 | 4.870694008658371 | 4386.170693968778 | 900.5227358096645 | random sphere packing, 200 grains |
| 11 | sphere_packing_800 | 800 | 3022 | 11 | 1.0 | 0.5 | 0.5 | 4.773906827240992 | 4759.045125358793 | 996.8868889946918 | random sphere packing, 800 grains |
| 12 | dense_random_contact_operator_256 | 32 | 256 | 256 | 7.2322647011819035 | 0.35 | 0.35 | 126.73385008833127 | 2889.3053912613395 | 22.798213652055427 | unstructured random contact operator, 256 contacts and 32 bodies, mu 0.35 |
| 13 | heavy_top_column_tangential_drive | 3 | 3 | 2 | 1000.0 | 0.5 | 0.5 | 0.0002927682519584372 | 2.0010002499999997 | 6834.758333987907 | three-body point-mass column with mass ratio 1000:1 and tangential drive on the heavy top body |
| 14 | stacked_mass_pairs_cone | 8 | 8 | 2 | 1000.0 | 0.5 | 0.5 | 7.602904288988335e-05 | 2.0017074190079107 | 26328.194370499747 | four point-mass pairs stacked with mass ratio 1000:1, Coulomb-cone contact |
| 15 | quadruped_sticking_stance | 1 | 4 | 4 | 5458.960759717582 | 0.5 | 0.5 | 0.016834848153442515 | 5.586013881554984 | 331.81254922175907 | floating-base quadruped stance (Unitree A1), four feet on plane, stick branch |
| 16 | quadruped_sliding_stance | 1 | 4 | 4 | 5458.960759717582 | 0.5 | 0.5 | 0.016834848153442515 | 5.586013881554984 | 331.81254922175907 | floating-base quadruped stance (Unitree A1), four feet on plane, slip branch (knee torque 3 Nm) |
| 17 | humanoid | 1 | 2 | 2 | 9711.641507237968 | 0.5 | 0.5 | 0.0578054382359583 | 0.5933695940732746 | 10.264944132958146 | floating-base humanoid standing, two feet on plane |
| 18 | quadruped_unloading_probe | 1 | 4 | 4 | 3153.4522739765475 | 0.5 | 0.5 | 0.2721286926465641 | 7.6100896191881535 | 27.965039427400633 | floating-base quadruped stance (Unitree A1), alternate contact configuration |
| 19 | batched_quadruped_sticking | 1 | 4 | 4 | 5472.157021330492 | 0.5 | 0.5 | 0.018988633444662107 | 5.586100304510231 | 294.1812701155985 | batched floating-base quadruped stance (Unitree A1), stick variant, batch element 0 |
| 20 | batched_quadruped_sliding | 1 | 4 | 4 | 5465.080406789735 | 0.5 | 0.5 | 0.01756848743372184 | 5.586491561418729 | 317.98363874488905 | batched floating-base quadruped stance (Unitree A1), slip variant, batch element 0 |
