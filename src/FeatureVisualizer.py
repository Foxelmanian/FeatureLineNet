import trimesh
import numpy as np
from mayavi import mlab
import os
import pandas as pd
import random

class FeatureVisualizer():
    def __init__(self, model, number_categories):
        self.__model = model
        self.__number_categories = number_categories

    def plot(self, stl_file_path, df):
        # Import the unlabeled file and use the model for category prediction

        X_pred = df[df.columns[6:]] # Feature values scalar products
        X_pred = np.array(X_pred)
        X_pred = np.reshape(X_pred, (len(X_pred), 6, 6, 1))


        point_coordinates = df[df.columns[:6]]  #  x y z coordinates of the points
        y_pred = self.__model.predict(X_pred)

        # Use the maximum probabillity as an estimation of the categorie
        if self.__number_categories > 2:
            print('Categorical plot')
            list_of_choosen_categorys_in_y = np.argmax(y_pred, axis=1)
        else:
            print('Not categorical plot')
            list_of_choosen_categorys_in_y = []
            c1, c2 = 0,0
            for yy in y_pred:
                if yy < 0.5:
                    c1 += 1
                    list_of_choosen_categorys_in_y.append(0)
                else:
                    c2 += 1
                    list_of_choosen_categorys_in_y.append(1)
            print(f'Categorie numbers: {c1},  {c2}')
            list_of_choosen_categorys_in_y = np.array(list_of_choosen_categorys_in_y).astype(int)

        triangle_mesh = trimesh.load(stl_file_path)
        point_sets = []

        new_data_frames = []
        # Build op a point set of each choosen list
        for categorie in range(self.__number_categories):
            true_category_list = []
            for categorie_choosen in list_of_choosen_categorys_in_y:
                if categorie == categorie_choosen:
                    true_category_list.append(True)
                else:
                    true_category_list.append(False)
            true_category_array = np.array(true_category_list)
            points = point_coordinates[true_category_array]
            print(f'Points {len(points)} of {len(true_category_list)}')
            point_sets.append(points)
            new_data_frames.append(df[true_category_array])

        # Plot the mesh and the point sets
        print(f"Number of categories found {len(point_sets)}")

        rgb_list = []
        for x in range(self.__number_categories):
            rgb_list.append((random.random(), random.random(), random.random()))

        plot_binary = True

        if not plot_binary:
            for point_set, color in zip(point_sets, rgb_list):
                print('Plot point set')
                engine = mlab.get_engine()
                mlab.figure(size=(1200, 1200), bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
                self.__plot_mesh(triangle_mesh)

                self.__plot_selected_points_on_mesh(triangle_mesh, point_set, color)
                engine._get_current_scene()
                mlab.show()
        else:
            p1, p2 = point_sets
            engine = mlab.get_engine()
            mlab.figure(size=(600, 600), bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
            self.__plot_mesh(triangle_mesh)
            #self.__plot_selected_points_on_mesh(triangle_mesh, p1, (220/255,220/255,220/255))
            #self.__plot_selected_points_on_mesh(triangle_mesh, p1, (0.25,0.25,0.25) )
            self.__plot_selected_points_on_mesh(triangle_mesh, p2, (1, 0, 0))
            #self.__plot_selected_lines(triangle_mesh, p2, (1,0,0) )

            # render_JPG = False
            #s = engine._get_current_scene()
            file_path = os.path.join('images')
            file_name = os.path.split(stl_file_path)[-1]
            file_name, extension = os.path.splitext(file_name)
            jpg_file = os.path.join(file_path, file_name + '.jpg')
            print(f'Save rendered picture to {jpg_file}')
            #s.scene.save(os.path.join(jpg_file))
            mlab.show()
            #mlab.close()

        return new_data_frames

    def __plot_mesh(self, triangle_mesh):
        mesh = mlab.triangular_mesh(triangle_mesh.vertices[:, 0],
                            triangle_mesh.vertices[:, 1],
                            triangle_mesh.vertices[:, 2],
                            triangle_mesh.faces, colormap="bone", opacity=1.0)
        lut = mesh.module_manager.scalar_lut_manager.lut.table.to_array()
        lut[:, 0] = 255
        lut[:, 1] = 255
        lut[:, 2] = 255
        mesh.module_manager.scalar_lut_manager.lut.table = lut

    def __plot_selected_points_on_mesh(self, triangle_mesh, points, point_color):
        x, y, z = [], [], []
        xp, yp, zp = [], [], []
        for xx1, xx2 in zip(points[points.columns[0]], points[points.columns[3]]):
            x.append([xx1, xx2])
            xp.append(xx1)
            xp.append(xx2)
        for yy1, yy2 in zip(points[points.columns[1]], points[points.columns[4]]):
            y.append([yy1, yy2])
            yp.append(yy1)
            yp.append(yy2)
        for zz1, zz2 in zip(points[points.columns[2]], points[points.columns[5]]):
            z.append([zz1, zz2])
            zp.append(zz1)
            zp.append(zz2)
        x_m = np.min(triangle_mesh.vertices[:, 0])
        y_m = np.min(triangle_mesh.vertices[:, 1])
        z_m = np.min(triangle_mesh.vertices[:, 2])
        x_ma = np.max(triangle_mesh.vertices[:, 0])
        y_ma = np.max(triangle_mesh.vertices[:, 1])
        z_ma = np.max(triangle_mesh.vertices[:, 2])
        scale_factor = 0.5 * 0.015 * (x_ma - x_m + y_ma + z_ma - z_m - y_m)
        mlab.points3d(xp, yp, zp, mode="sphere", color=point_color, scale_factor=0.5 * scale_factor, opacity=1.0)


    def __plot_selected_lines(self, triangle_mesh, points, point_color):
        x, y, z = [], [], []
        for xx1, xx2 in zip(points[points.columns[0]], points[points.columns[3]]):
            x.append([xx1, xx2])
        for yy1, yy2 in zip(points[points.columns[1]], points[points.columns[4]]):
            y.append([yy1, yy2])
        for zz1, zz2 in zip(points[points.columns[2]], points[points.columns[5]]):
            z.append([zz1, zz2])
        x_m = np.min(triangle_mesh.vertices[:, 0])
        y_m = np.min(triangle_mesh.vertices[:, 1])
        z_m = np.min(triangle_mesh.vertices[:, 2])
        x_ma = np.max(triangle_mesh.vertices[:, 0])
        y_ma = np.max(triangle_mesh.vertices[:, 1])
        z_ma = np.max(triangle_mesh.vertices[:, 2])
        scale_factor = 0.5 * 0.015 * (x_ma - x_m + y_ma + z_ma - z_m - y_m)
        for x_l, y_l, z_l in zip(x,y,z):
            mlab.plot3d(x_l,y_l,z_l, color=point_color, tube_radius=0.25 *scale_factor,opacity=1.0 )
