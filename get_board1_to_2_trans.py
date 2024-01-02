import cv2
import cv2.aruco as aruco
import numpy as np
from detector import parse_cam_param, gmm_fit, LdmkProjector
from utils.mesh import TriMesh
import estimate_rigid_transform as ert
import os
from tqdm import tqdm

def dict_to_np(dict):
    id_list = []
    np_list = []
    for id, points in dict.items():
        id_list.append(id)
    # sort id
    id_list = np.array(id_list)
    id_list = np.sort(id_list)
    for id in id_list:
        for point in dict[id]:
            np_list.append(point)
    return np.array(np_list)

def gen_board():
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
    return board1, board2, dictionary


def get_board_points_pos_dict(board, dictionary, mesh_path, image_dir,cameras_path):

    img_names = sorted([os.path.splitext(filename)[0] for filename in os.listdir(image_dir)])

    cam_mat_lst, trans_mat_lst, valid_lst = parse_cam_param(cameras_path, img_names)

    res_list = []
    res_ids = None
    valid_cam_lst = []
    valid_trans_lst = []
    #add process bar
    print("identifying charuco corners")
    progress_bar = tqdm(enumerate(img_names), total=len(img_names), unit="%", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}]")


    for i, img_name  in progress_bar:
        img_path = image_dir + "/" +img_name + ".png"
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        markerCorners1, markerIds1, rejectedImgPoints1 = cv2.aruco.detectMarkers(img, dictionary)
        if len(markerCorners1) == 0:
            continue
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(markerCorners1, markerIds1, gray, board)
        if retval == 0:
            continue
        res_list.append(charuco_corners[:,0,:])
        if res_ids is None:
            res_ids = charuco_ids
        else:
            res_ids = np.vstack((res_ids,charuco_ids))
        valid_cam_lst.append(cam_mat_lst[i])
        valid_trans_lst.append(trans_mat_lst[i])

        progress_percent = (i + 1) / len(img_names) * 100

        # 更新进度条
        progress_bar.set_postfix(progress="{:.2f}%".format(progress_percent))
        progress_bar.update(1)
    mesh = TriMesh()
    mesh.load(mesh_path)

    projector = LdmkProjector()
    for i in range(len(res_list)):
        lmk_3d = projector.ldmk_3d_detect(mesh, [res_list[i]], [valid_cam_lst[i]], [valid_trans_lst[i]])
        if i == 0 :
            lmk_3d_list = lmk_3d
        else:
            lmk_3d_list = np.vstack((lmk_3d_list,lmk_3d))
        
        # 创建一个空字典用于存储相同ID的点
    points_dict = {}

    # 遍历res_ids和lmk_3d_list，将相同ID的点组合成子列表
    for id, point in zip(res_ids, lmk_3d_list):
        if int(id) not in points_dict:
            points_dict[int(id)] = []  # 初始化空子列表
        points_dict[int(id)].append(point)  # 将点添加到相应的子列表中

    # for id, points in points_dict.items():
    #     points_dict[id] = gmm_fit(np.array(points))

    # return points_dict
    return points_dict

def get_board_points_mean_pos_dict(board, dictionary, mesh_path, image_dir,cameras_path):

    img_names = sorted([os.path.splitext(filename)[0] for filename in os.listdir(image_dir)])

    cam_mat_lst, trans_mat_lst, valid_lst = parse_cam_param(cameras_path, img_names)

    res_list = []
    res_ids = None
    valid_cam_lst = []
    valid_trans_lst = []
    #add process bar
    print("identifying charuco corners")
    progress_bar = tqdm(enumerate(img_names), total=len(img_names), unit="%", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}]")


    for i, img_name  in progress_bar:
        img_path = image_dir + "/" +img_name + ".png"
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        markerCorners1, markerIds1, rejectedImgPoints1 = cv2.aruco.detectMarkers(img, dictionary)
        if len(markerCorners1) == 0:
            continue
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(markerCorners1, markerIds1, gray, board)
        if retval == 0:
            continue
        res_list.append(charuco_corners[:,0,:])
        if res_ids is None:
            res_ids = charuco_ids
        else:
            res_ids = np.vstack((res_ids,charuco_ids))
        valid_cam_lst.append(cam_mat_lst[i])
        valid_trans_lst.append(trans_mat_lst[i])

        progress_percent = (i + 1) / len(img_names) * 100

        # 更新进度条
        progress_bar.set_postfix(progress="{:.2f}%".format(progress_percent))
        progress_bar.update(1)
    mesh = TriMesh()
    mesh.load(mesh_path)

    projector = LdmkProjector()
    print("projecting 3d landmark")
    for i in range(len(res_list)):
        lmk_3d = projector.ldmk_3d_detect(mesh, [res_list[i]], [valid_cam_lst[i]], [valid_trans_lst[i]])
        if i == 0 :
            lmk_3d_list = lmk_3d
        else:
            lmk_3d_list = np.vstack((lmk_3d_list,lmk_3d))
        
        # 创建一个空字典用于存储相同ID的点
    points_dict = {}

    # 遍历res_ids和lmk_3d_list，将相同ID的点组合成子列表
    for id, point in zip(res_ids, lmk_3d_list):
        if int(id) not in points_dict:
            points_dict[int(id)] = []  # 初始化空子列表
        points_dict[int(id)].append(point)  # 将点添加到相应的子列表中

    for id, points in points_dict.items():
        points_dict[id] = gmm_fit(np.array(points))

    return points_dict

def get_world_board1_to_2_trans_old(mesh_path,image_dir,cameras_path,board1,board2,dictionary):
    print("processing board1")
    board1_points_dict = get_board_points_mean_pos_dict(board1, dictionary, mesh_path, image_dir, cameras_path)
    print("processing board2")
    board2_points_dict = get_board_points_mean_pos_dict(board2, dictionary, mesh_path, image_dir, cameras_path)
    # 取出两个字典中相同ID的点，形成两个新列表，一一对应
    print("get same id points")
    board1_points = []
    board2_points = []
    for id in board1_points_dict:
        if id in board2_points_dict:
            board1_points.append(board1_points_dict[id])
            board2_points.append(board2_points_dict[id])

    # 将两个列表转换为数组
    board1_points = np.array(board1_points)
    board2_points = np.array(board2_points)
    print("estimate rigid transform")
    M = ert.affine_matrix_from_points(board1_points.transpose(), board2_points.transpose(), scale=True)
    return M

def gen_template_space(board):
    dictionary = board.getDictionary()
    img = board.generateImage((1536, 2048), marginSize=0)
    cv2.imwrite("template.png", img)
    img = cv2.imread("template.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    markerCorners1, markerIds1, rejectedImgPoints1 = cv2.aruco.detectMarkers(img, dictionary)
    retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(markerCorners1, markerIds1, gray, board)
    # trans
    center_x = np. mean(charuco_corners[:,0,0])
    center_y = np. mean(charuco_corners[:,0,1])
    charuco_corners[:,0,0] = charuco_corners[:,0,0] - center_x
    charuco_corners[:,0,1] = charuco_corners[:,0,1] - center_y
    # scale
    scale = 0.025 / 512
    charuco_corners[:,0,0] = charuco_corners[:,0,0] * scale
    charuco_corners[:,0,1] = charuco_corners[:,0,1] * scale
    # add z axis 
    charuco_corners = charuco_corners[:,0,:]
    temp_world_coord = np.hstack((charuco_corners, np.zeros((len(charuco_corners), 1))))
    # temp_world_coord and ids dict
    points_dict = {}

    for id, point in zip(charuco_ids[:,0], temp_world_coord):
        if int(id) not in points_dict:
            points_dict[int(id)] = []  # 初始化空子列表
        points_dict[int(id)].append(point)  # 将点添加到相应的子列表中
        

    return points_dict

def add_cut_template(temp_world_dict,board1_points_dict):
    keys_to_delete = []
    for id, points in temp_world_dict.items():
        if id not in board1_points_dict:
            keys_to_delete.append(id)
        else:
            len_key = len(board1_points_dict[id])
            # repeat temp_world_dict[id] to len_key
            temp_world_dict[id] = np.tile(temp_world_dict[id][0],(len_key,1))
    for key in keys_to_delete:
        del temp_world_dict[key]

    return temp_world_dict 

def get_board_to_temp_M(board1_points_dict,board1):
    temp_world_dict = gen_template_space(board1)
    temp_world_dict = add_cut_template(temp_world_dict,board1_points_dict)
    board1_points = dict_to_np(board1_points_dict)
    temp_world_points = dict_to_np(temp_world_dict)
    M = ert.affine_matrix_from_points(board1_points.transpose(), temp_world_points.transpose(), scale=True)
    return M

def get_board1_to_2_trans(board1_points_dict, board2_points_dict,board1,board2):
    M1 = get_board_to_temp_M(board1_points_dict,board1)
    M2 = get_board_to_temp_M(board2_points_dict,board2)
    M = np.matmul(np.linalg.inv(M2),M1)
    return M


if __name__ == "__main__":
    board1, board2, dictionary = gen_board()
    exp = "align_merge_out_2023_12_07_90degree_ldr"
    mesh_path ='/home/yuruihan/DS-FaceScape/hdr_emitter/metashape_output/{}/model.obj'.format(exp)
    image_dir = "/home/yuruihan/DS-FaceScape/hdr_emitter/{}".format(exp)
    cameras_path = '/home/yuruihan/DS-FaceScape/hdr_emitter/metashape_output/{}/camera.xml'.format(exp)
    M = get_world_board1_to_2_trans_old(mesh_path,image_dir,cameras_path,board1,board2,dictionary)

    board1_points_dict = get_board_points_pos_dict(board1, dictionary, mesh_path, image_dir, cameras_path)
    board2_points_dict = get_board_points_pos_dict(board2, dictionary, mesh_path, image_dir, cameras_path)
    import ipdb;ipdb.set_trace()
    M_prim = get_board1_to_2_trans(board1_points_dict, board2_points_dict,board1,board2)
    import ipdb;ipdb.set_trace()
    # print(M)
    
    