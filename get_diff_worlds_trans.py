from get_board1_to_2_trans import gen_board, get_board1_to_2_trans, get_board_points_pos_dict,get_board_to_temp_M
import numpy as np
import estimate_rigid_transform as ert

def get_diff_worlds_trans(board1_points_dict, board2_points_dict,board1,board2):
    M1 = get_board_to_temp_M(board1_points_dict,board1)
    M2 = get_board_to_temp_M(board2_points_dict,board2)
    M = np.matmul(np.linalg.inv(M2),M1)
    return M

# def get_diff_worlds_trans(world1_mesh_path, world1_image_dir, world1_cameras_path,world2_mesh_path, world2_image_dir, world2_cameras_path,board1, dictionary):
#     world1_points_dict = get_board_points_mean_pos_dict(board1, dictionary, world1_mesh_path, world1_image_dir, world1_cameras_path)
#     world2_points_dict = get_board_points_mean_pos_dict(board1, dictionary, world2_mesh_path, world2_image_dir, world2_cameras_path)
#     world1_points = []
#     world2_points = []
#     for id in world1_points_dict:
#         if id in world2_points_dict:
#             world1_points .append(world1_points_dict[id])
#             world2_points.append(world2_points_dict[id])

#     world1_points  = np.array(world1_points )
#     world2_points = np.array(world2_points)
#     M = ert.affine_matrix_from_points(world1_points.transpose(), world2_points.transpose(), scale=True)
#     # M represent world1 to world2
#     return M

# def get_all_trans(worldlist,parent_dir):
#     world_mesh_path = []
#     world_image_dir = []
#     world_cameras_path = []
#     for world in worldlist:
#         world_mesh_path.append(parent_dir +  "/metashape_output/" + world + "/model.obj")
#         world_image_dir.append(parent_dir +"/"+ world)
#         world_cameras_path.append(parent_dir +  "/metashape_output/" + world +  "/camera.xml")
#     board1, board2, dictionary = gen_board()
#     print("get world trans")
#     world_trans21 = get_diff_worlds_trans(world_mesh_path[1], world_image_dir[1], world_cameras_path[1],world_mesh_path[0], world_image_dir[0], world_cameras_path[0],board1, dictionary)
#     print("world2 to world1 trans has got")
#     world_trans31 = get_diff_worlds_trans(world_mesh_path[2], world_image_dir[2], world_cameras_path[2],world_mesh_path[0], world_image_dir[0], world_cameras_path[0],board1, dictionary)
#     print("world3 to world1 trans has got")
#     world_trans41 = get_diff_worlds_trans(world_mesh_path[3], world_image_dir[3], world_cameras_path[3],world_mesh_path[0], world_image_dir[0], world_cameras_path[0],board1, dictionary)
#     print("world4 to world1 trans has got")
#     print("get board trans in each world")
#     world1_board1_to_2 = get_world_board1_to_2_trans(world_mesh_path[0], world_image_dir[0], world_cameras_path[0],board1,board2,dictionary)
#     print("world1_board1_to_2 has got")
#     world2_board1_to_2 = get_world_board1_to_2_trans(world_mesh_path[1], world_image_dir[1], world_cameras_path[1],board1,board2,dictionary)
#     print("world2_board1_to_2 has got")
#     world3_board1_to_2 = get_world_board1_to_2_trans(world_mesh_path[2], world_image_dir[2], world_cameras_path[2],board1,board2,dictionary)
#     print("world3_board1_to_2 has got")
#     world4_board1_to_2 = get_world_board1_to_2_trans(world_mesh_path[3], world_image_dir[3], world_cameras_path[3],board1,board2,dictionary)
#     print("world4_board1_to_2 has got")
#     world_trans_list = [world_trans21,world_trans31,world_trans41]
#     world_board1_to_2_list = [world1_board1_to_2,world2_board1_to_2,world3_board1_to_2,world4_board1_to_2]

#     return world_trans_list, world_board1_to_2_list


def get_all_trans(worldlist,parent_dir):
    world_mesh_path = []
    world_image_dir = []
    world_cameras_path = []
    for world in worldlist:
        world_mesh_path.append(parent_dir +  "/metashape_output/" + world + "/model.obj")
        world_image_dir.append(parent_dir +"/"+ world)
        world_cameras_path.append(parent_dir +  "/metashape_output/" + world +  "/camera.xml")
    board1, board2, dictionary = gen_board()
    print("get point dict")
    print("get board1 points dict")
    world1_board1_points_dict = get_board_points_pos_dict(board1, dictionary, world_mesh_path[0], world_image_dir[0], world_cameras_path[0])
    
    world2_board1_points_dict = get_board_points_pos_dict(board1, dictionary, world_mesh_path[1], world_image_dir[1], world_cameras_path[1])
    world3_board1_points_dict = get_board_points_pos_dict(board1, dictionary, world_mesh_path[2], world_image_dir[2], world_cameras_path[2])
    world4_board1_points_dict = get_board_points_pos_dict(board1, dictionary, world_mesh_path[3], world_image_dir[3], world_cameras_path[3])
    print("get board2 points dict")
    world1_board2_points_dict = get_board_points_pos_dict(board2, dictionary, world_mesh_path[0], world_image_dir[0], world_cameras_path[0])
    world2_board2_points_dict = get_board_points_pos_dict(board2, dictionary, world_mesh_path[1], world_image_dir[1], world_cameras_path[1])
    world3_board2_points_dict = get_board_points_pos_dict(board2, dictionary, world_mesh_path[2], world_image_dir[2], world_cameras_path[2])
    world4_board2_points_dict = get_board_points_pos_dict(board2, dictionary, world_mesh_path[3], world_image_dir[3], world_cameras_path[3])
    print("get world trans")
    world_trans21 = get_diff_worlds_trans(world2_board1_points_dict, world1_board1_points_dict,board1,board1)
    world_trans31 = get_diff_worlds_trans(world3_board1_points_dict, world1_board1_points_dict,board1,board1)
    world_trans41 = get_diff_worlds_trans(world4_board1_points_dict, world1_board1_points_dict,board1,board1)
    print("get board trans in each world")
    world1_board1_to_2 = get_board1_to_2_trans(world1_board1_points_dict, world1_board2_points_dict,board1,board2)
    world2_board1_to_2 = get_board1_to_2_trans(world2_board1_points_dict, world2_board2_points_dict,board1,board2)
    world3_board1_to_2 = get_board1_to_2_trans(world3_board1_points_dict, world3_board2_points_dict,board1,board2)
    world4_board1_to_2 = get_board1_to_2_trans(world4_board1_points_dict, world4_board2_points_dict,board1,board2)


    
    
    
    
    # print("get world trans")
    # world_trans21 = get_diff_worlds_trans(world_mesh_path[1], world_image_dir[1], world_cameras_path[1],world_mesh_path[0], world_image_dir[0], world_cameras_path[0],board1, dictionary)
    # print("world2 to world1 trans has got")
    # world_trans31 = get_diff_worlds_trans(world_mesh_path[2], world_image_dir[2], world_cameras_path[2],world_mesh_path[0], world_image_dir[0], world_cameras_path[0],board1, dictionary)
    # print("world3 to world1 trans has got")
    # world_trans41 = get_diff_worlds_trans(world_mesh_path[3], world_image_dir[3], world_cameras_path[3],world_mesh_path[0], world_image_dir[0], world_cameras_path[0],board1, dictionary)
    # print("world4 to world1 trans has got")
    # print("get board trans in each world")
    # world1_board1_to_2 = get_world_board1_to_2_trans(world_mesh_path[0], world_image_dir[0], world_cameras_path[0],board1,board2,dictionary)
    # print("world1_board1_to_2 has got")
    # world2_board1_to_2 = get_world_board1_to_2_trans(world_mesh_path[1], world_image_dir[1], world_cameras_path[1],board1,board2,dictionary)
    # print("world2_board1_to_2 has got")
    # world3_board1_to_2 = get_world_board1_to_2_trans(world_mesh_path[2], world_image_dir[2], world_cameras_path[2],board1,board2,dictionary)
    # print("world3_board1_to_2 has got")
    # world4_board1_to_2 = get_world_board1_to_2_trans(world_mesh_path[3], world_image_dir[3], world_cameras_path[3],board1,board2,dictionary)
    # print("world4_board1_to_2 has got")
    world_trans_list = [world_trans21,world_trans31,world_trans41]
    world_board1_to_2_list = [world1_board1_to_2,world2_board1_to_2,world3_board1_to_2,world4_board1_to_2]

    world_trans10 = get_board_to_temp_M(world1_board1_points_dict,board1)

    # world_trans_list = [world_trans21]
    # world_board1_to_2_list = [world1_board1_to_2]

    return world_trans_list, world_board1_to_2_list,world_trans10

if __name__ == "__main__":
    world_keyword_lst =["degree","90degree","180degree","270degree"]
    worldlist = []
    for world_keyword in world_keyword_lst:
        worldlist.append("align_merge_out_2023_12_07_{}_ldr".format(world_keyword))
    parent_dir = "/home/yuruihan/DS-FaceScape/hdr_emitter"

    world_trans_list, world_board1_to_2_list,world_trans10 = get_all_trans(worldlist,parent_dir)

    output_file = "all_trans.txt"
    # create empty txt
    with open(output_file, 'w') as file:
        file.write('')

    for world_trans in world_trans_list:
        with open(output_file, 'a') as file:
            np.savetxt(file, world_trans, delimiter=',', fmt='%f')
            file.write('\n')
    for board1_to_2_array in world_board1_to_2_list:
        with open(output_file, 'a') as file:
            np.savetxt(file, board1_to_2_array, delimiter=',', fmt='%f')
            file.write('\n')
    print(world_trans10)
    with open(output_file, 'a') as file:
        np.savetxt(file, world_trans10, delimiter=',', fmt='%f')
        file.write('\n')


