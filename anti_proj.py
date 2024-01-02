
from utils.octree import OCTree
from utils.mesh import TriMesh
from utils.correspondences import Correspondences
from sklearn.mixture import GaussianMixture
import numpy as np
from detector import LdmkProjector
import cv2

import os
#2d points to 3d points
def p2d2pd3(board_points, mesh_path, cam_mat_lst, trans_mat_lst): 
    res_list = np.array(board_points)
    res_list = res_list[:,0,:]

    mesh = TriMesh()
    mesh.load(mesh_path)

    projector = LdmkProjector()
    for i in range(len([res_list])):
        lmk_3d = projector.ldmk_3d_detect(mesh, [res_list], [cam_mat_lst[i]], [trans_mat_lst[i]])
        if i == 0 :
            lmk_3d_list = lmk_3d
        else:
            lmk_3d_list = np.vstack((lmk_3d_list,lmk_3d))
    return lmk_3d_list

def transform_mat2vec(trans_mat):
    R = trans_mat[:3, :3]
    t = trans_mat[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    return rvec,t


#3d points to 2d points
def get_antiporj(transformed_lmk_3d_list,cam_mat,trans_mat):
    rvec,tvec = transform_mat2vec(trans_mat)
    board1_transformed_points_2d, _ = cv2.projectPoints(transformed_lmk_3d_list, rvec,tvec, cam_mat, distCoeffs=None)



if __name__  == "__main__":
    from detector import parse_cam_param, gmm_fit, LdmkProjector

    import cv2
    import cv2.aruco as aruco
    import numpy as np

    exps = ["align_merge_out_2023_12_07_degree_ldr","align_merge_out_2023_12_07_90degree_ldr","align_merge_out_2023_12_07_180degree_ldr","align_merge_out_2023_12_07_270degree_ldr"]

    for exp in exps:
        print("begin to process {}".format(exp))
        # exp = "align_merge_out_2023_12_07_90degree_ldr"
        mesh_path ='/home/yuruihan/DS-FaceScape/hdr_emitter/metashape_output/{}/model.obj'.format(exp)
        image_dir = "/home/yuruihan/DS-FaceScape/hdr_emitter/{}".format(exp)
        cameras_path = '/home/yuruihan/DS-FaceScape/hdr_emitter/metashape_output/{}/camera.xml'.format(exp)



        # Define parameters for the CharucoBoard
        num_squares_x = 7
        num_squares_y = 10
        square_length = 0.04  # length of each square side in meters
        marker_length = 0.02  # length of the markers in meters
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)  # you can choose a different dictionary

        # Define a nonzero start ID for aruco markers
        start_id = 200

        # Create CharucoBoard with a nonzero start ID
        board1 = aruco.CharucoBoard(
            (num_squares_x, num_squares_y),
            squareLength=square_length,
            markerLength=marker_length,
            dictionary=dictionary,
            ids=np.arange(start_id, start_id+num_squares_x*num_squares_y//2, dtype=np.int32)
        )

        board2 = aruco.CharucoBoard(
            (num_squares_x, num_squares_y),
            squareLength=square_length,
            markerLength=marker_length,
            dictionary=dictionary,
            ids=board1.getIds() + len(board1.getIds()),
        )

            
        img_names = sorted([os.path.splitext(filename)[0] for filename in os.listdir(image_dir)])

        all = []

        for i, img_name in enumerate(img_names):
            print("process {}th image".format(i))
            if '.jpg' in img_name or '.JPG' in img_name or '.png' in img_name or '.PNG' in img_name:
                img_name = img_name[:-4]
            img_path = image_dir + "/" +img_name + ".png"
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            markerCorners1, markerIds1, rejectedImgPoints1 = cv2.aruco.detectMarkers(img, dictionary)
            if len(markerCorners1) == 0:
                continue
            retval, test_charuco_corners1, test_charuco_ids1 = aruco.interpolateCornersCharuco(markerCorners1, markerIds1, gray, board1)
            if retval == 0:
                continue
            retval, test_charuco_corners2, test_charuco_ids2 = aruco.interpolateCornersCharuco(markerCorners1, markerIds1, gray, board2)
            if retval == 0:
                continue
            #  # find the same id in board1 and board2
            # test_board1_points = []
            # test_board2_points = []
            # for id in test_charuco_ids1:
            #     if id in test_charuco_ids2:
            #         # get the index of the same id
            #         index1 = np.argwhere(test_charuco_ids1 == id)[0][0]
            #         index2 = np.argwhere(test_charuco_ids2 == id)[0][0]
            #         # get the points of the same id
            #         test_board1_points.append(test_charuco_corners1[index1])
            #         test_board2_points.append(test_charuco_corners2[index2])
            # if len(test_board1_points) == 0:
            #     continue

            # cat all the points
            test_board_points = np.vstack((test_charuco_corners1,test_charuco_corners2))

            cam_mat_lst, trans_mat_lst, valid_lst = parse_cam_param(cameras_path, [img_name])
            board1_lmk_3d_list = p2d2pd3(test_board_points,mesh_path, cam_mat_lst, trans_mat_lst)
            
            # save as obj
            # Save as OBJ file
            output_file = '/home/yuruihan/DS-FaceScape/hdr_emitter/metashape_output/{}/pc/{}.obj'.format(exp, img_name)
            # create dir
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            print("save as {}".format(output_file))
            with open(output_file, 'w') as f:
                for point in board1_lmk_3d_list:
                    all.append(point)
                    f.write(f"v {point[0]} {point[1]} {point[2]}\n")
        print("save all point of {}".format(exp))
        with open('/home/yuruihan/DS-FaceScape/hdr_emitter/metashape_output/{}/pc/all.obj'.format(exp), 'w') as f:
            for point in all:
                f.write(f"v {point[0]} {point[1]} {point[2]}\n")

