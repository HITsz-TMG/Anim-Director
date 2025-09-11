import sys
import io
import os
import cv2
import math
import numpy as np
from scipy.signal import medfilt
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
from PIL import Image

from dust3r.inference import inference

from dust3r.utils.image import load_images# , resize_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

from third_party.utils.camera_utils import remap_points
from third_party.utils.img_utils import rgba_to_rgb, resize_with_aspect_ratio
from third_party.utils.img_utils import compute_img_diff

from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def suppress_output(func):
    def wrapper(*args, **kwargs):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    return wrapper

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = Image.LANCZOS
    elif S <= long_edge_size:
        interp = Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def resize_images(imgs_list, size, square_ok=False):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    imgs = []
    for img in imgs_list:
        img = exif_transpose(Image.fromarray(img)).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    return imgs

@suppress_output
def infer_match(images, model, vis=False, niter=300, lr=0.01, schedule='cosine', device="cuda:0"):
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    
    images_packed = resize_images(images, size=512, square_ok=True)
    # images_packed = images

    pairs = make_pairs(images_packed, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=False)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    # focals = scene.get_focals()
    # poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    # visualize reconstruction
    # scene.show()

    # find 2D-2D matches between the two images
    pts2d_list, pts3d_list = [], []
    for i in range(2):
        conf_i = confidence_masks[i].cpu().numpy()
        pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
        pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
        if pts3d_list[-1].shape[0] == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))
    reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)

    matches_im1 = pts2d_list[1][reciprocal_in_P2]
    matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    # visualize a few matches
    if vis == True:
        print(f'found {num_matches} matches')
        n_viz = 20
        match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
        img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = np.concatenate((img0, img1), axis=1)
        plt.figure()
        plt.imshow(img)
        cmap = plt.get_cmap('jet')
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        plt.show(block=True)

    matches_im1 = remap_points(images[1].shape, matches_im1)
    return matches_im0, matches_im1


def point_transform(H, pt):
    """
    @param: H is homography matrix of dimension (3x3)
    @param: pt is the (x, y) point to be transformed

    Return:
            returns a transformed point ptrans = H*pt.
    """
    a = H[0, 0] * pt[0] + H[0, 1] * pt[1] + H[0, 2]
    b = H[1, 0] * pt[0] + H[1, 1] * pt[1] + H[1, 2]
    c = H[2, 0] * pt[0] + H[2, 1] * pt[1] + H[2, 2]
    return [a / c, b / c]


def points_transform(H, pt_x, pt_y):
    """
    @param: H is homography matrix of dimension (3x3)
    @param: pt is the (x, y) point to be transformed

    Return:
            returns a transformed point ptrans = H*pt.
    """
    a = H[0, 0] * pt_x + H[0, 1] * pt_y + H[0, 2]
    b = H[1, 0] * pt_x + H[1, 1] * pt_y + H[1, 2]
    c = H[2, 0] * pt_x + H[2, 1] * pt_y + H[2, 2]
    return (a / c, b / c)


def motion_propagate(old_points, new_points, old_size, new_size, H_size=(21, 21)):
    """
    @param: old_points are points in old_frame that are
            matched feature points with new_frame
    @param: new_points are points in new_frame that are
            matched feature points with old_frame
    @param: old_frame is the frame to which
            motion mesh needs to be obtained
    @param: H is the homography between old and new points

    Return:
            returns a motion mesh in x-direction
            and y-direction for old_frame
    """
    # spreads motion over the mesh for the old_frame
    x_motion = np.zeros(H_size)
    y_motion = np.zeros(H_size)
    mesh_x_num, mesh_y_num = H_size[0], H_size[1]
    pixels_x, pixels_y = (old_size[1]) / (mesh_x_num - 1), (old_size[0]) / (mesh_y_num - 1)
    radius = max(pixels_x, pixels_y) * 5
    sigma = radius / 3.0

    H_global = None
    if old_points.shape[0] > 3:
        # pre-warping with global homography
        H_global, _ = cv2.findHomography(old_points, new_points, cv2.RANSAC)
    if H_global is None:
        old_tmp = np.array([[0, 0], [0, old_size[0]], [old_size[1], 0], [old_size[1], old_size[0]]])
        new_tmp = np.array([[0, 0], [0, new_size[0]], [new_size[1], 0], [new_size[1], new_size[0]]])
        H_global, _ = cv2.findHomography(old_tmp, new_tmp, cv2.RANSAC)

    for i in range(mesh_x_num):
        for j in range(mesh_y_num):
            pt = [pixels_x * i, pixels_y * j]
            ptrans = point_transform(H_global, pt)
            x_motion[i, j] = ptrans[0]
            y_motion[i, j] = ptrans[1]

    # disturbute feature motion vectors
    weighted_move_x = np.zeros(H_size)
    weighted_move_y = np.zeros(H_size)
    # 构建 KDTree
    tree = KDTree(old_points)
    # 计算权重和移动值
    for i in range(mesh_x_num):
        for j in range(mesh_y_num):
            vertex = [pixels_x * i, pixels_y * j]
            neighbor_indices = tree.query_ball_point(vertex, radius, workers=-1)
            if len(neighbor_indices) > 0:
                pts = old_points[neighbor_indices]
                sts = new_points[neighbor_indices]
                ptrans_x, ptrans_y = points_transform(H_global, pts[:, 0], pts[:, 1])
                moves_x = sts[:, 0] - ptrans_x
                moves_y = sts[:, 1] - ptrans_y

                dists = np.sqrt((vertex[0] - pts[:, 0]) ** 2 + (vertex[1] - pts[:, 1]) ** 2)
                weights_x = np.exp(-(dists ** 2) / (2 * sigma ** 2))
                weights_y = np.exp(-(dists ** 2) / (2 * sigma ** 2))

                weighted_move_x[i, j] = np.sum(weights_x * moves_x) / (np.sum(weights_x) + 0.1)
                weighted_move_y[i, j] = np.sum(weights_y * moves_y) / (np.sum(weights_y) + 0.1)

    x_motion_mesh = x_motion + weighted_move_x
    y_motion_mesh = y_motion + weighted_move_y
    '''
    # apply median filter (f-1) on obtained motion for each vertex
    x_motion_mesh = np.zeros((mesh_x_num, mesh_y_num), dtype=float)
    y_motion_mesh = np.zeros((mesh_x_num, mesh_y_num), dtype=float)

    for key in x_motion.keys():
        try:
            temp_x_motion[key].sort()
            x_motion_mesh[key] = x_motion[key]+temp_x_motion[key][len(temp_x_motion[key])//2]
        except KeyError:
            x_motion_mesh[key] = x_motion[key]
        try:
            temp_y_motion[key].sort()
            y_motion_mesh[key] = y_motion[key]+temp_y_motion[key][len(temp_y_motion[key])//2]
        except KeyError:
            y_motion_mesh[key] = y_motion[key]

    # apply second median filter (f-2) over the motion mesh for outliers
    #x_motion_mesh = medfilt(x_motion_mesh, kernel_size=[3, 3])
    #y_motion_mesh = medfilt(y_motion_mesh, kernel_size=[3, 3])
    '''
    return x_motion_mesh, y_motion_mesh


def mesh_warp_points(points, x_motion_mesh, y_motion_mesh, img_size):
    ptrans = []
    mesh_x_num, mesh_y_num = x_motion_mesh.shape
    pixels_x, pixels_y = (img_size[1]) / (mesh_x_num - 1), (img_size[0]) / (mesh_y_num - 1)
    for pt in points:
        i = int(pt[0] // pixels_x)
        j = int(pt[1] // pixels_y)
        src = [[i * pixels_x, j * pixels_y],
               [(i + 1) * pixels_x, j * pixels_y],
               [i * pixels_x, (j + 1) * pixels_y],
               [(i + 1) * pixels_x, (j + 1) * pixels_y]]
        src = np.asarray(src)
        dst = [[x_motion_mesh[i, j], y_motion_mesh[i, j]],
               [x_motion_mesh[i + 1, j], y_motion_mesh[i + 1, j]],
               [x_motion_mesh[i, j + 1], y_motion_mesh[i, j + 1]],
               [x_motion_mesh[i + 1, j + 1], y_motion_mesh[i + 1, j + 1]]]
        dst = np.asarray(dst)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
        x, y = points_transform(H, pt[0], pt[1])
        ptrans.append([x, y])

    return np.array(ptrans)


def mesh_warp_frame(frame, x_motion_mesh, y_motion_mesh, resize):
    """
    @param: frame is the current frame
    @param: x_motion_mesh is the motion_mesh to
            be warped on frame along x-direction
    @param: y_motion_mesh is the motion mesh to
            be warped on frame along y-direction
    @param: resize is the desired output size (tuple of width, height)

    Returns:
            returns a mesh warped frame according
            to given motion meshes x_motion_mesh,
            y_motion_mesh, resized to the specified size
    """

    map_x = np.zeros(resize, np.float32)
    map_y = np.zeros(resize, np.float32)

    mesh_x_num, mesh_y_num = x_motion_mesh.shape
    pixels_x, pixels_y = (resize[1]) / (mesh_x_num - 1), (resize[0]) / (mesh_y_num - 1)

    for i in range(mesh_x_num - 1):
        for j in range(mesh_y_num - 1):
            src = [[i * pixels_x, j * pixels_y],
                   [(i + 1) * pixels_x, j * pixels_y],
                   [i * pixels_x, (j + 1) * pixels_y],
                   [(i + 1) * pixels_x, (j + 1) * pixels_y]]
            src = np.asarray(src)

            dst = [[x_motion_mesh[i, j], y_motion_mesh[i, j]],
                   [x_motion_mesh[i + 1, j], y_motion_mesh[i + 1, j]],
                   [x_motion_mesh[i, j + 1], y_motion_mesh[i, j + 1]],
                   [x_motion_mesh[i + 1, j + 1], y_motion_mesh[i + 1, j + 1]]]
            dst = np.asarray(dst)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC)

            start_x = math.ceil(pixels_x * i)
            end_x = math.ceil(pixels_x * (i + 1))
            start_y = math.ceil(pixels_y * j)
            end_y = math.ceil(pixels_y * (j + 1))

            x, y = np.meshgrid(range(start_x, end_x), range(start_y, end_y), indexing='ij')

            map_x[y, x], map_y[y, x] = points_transform(H, x, y)

    # deforms mesh and directly outputs the resized frame
    resized_frame = cv2.remap(frame, map_x, map_y,
                              interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))
    return resized_frame


def infer_warp_mesh_img(src, dst, model, vis=False):
    if isinstance(src, str):
        image1 = cv2.imread(src,   cv2.IMREAD_UNCHANGED)
        image2 = cv2.imread(dst, cv2.IMREAD_UNCHANGED)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    elif isinstance(src, Image.Image):
        image1 = np.array(src)
        image2 = np.array(dst)
    else:
        assert isinstance(src, np.ndarray)

    image1 = rgba_to_rgb(image1)
    image2 = rgba_to_rgb(image2)

    image1_padded = resize_with_aspect_ratio(image1, image2)
    resized_image1 = cv2.resize(image1_padded, (image2.shape[1], image2.shape[0]), interpolation=cv2.INTER_AREA)

    matches_im0, matches_im1 = infer_match([resized_image1, image2], model, vis=vis)
    matches_im0 = matches_im0 * image1_padded.shape[0] / resized_image1.shape[0]

    # print('Estimate Mesh Grid')
    mesh_x, mesh_y = motion_propagate(matches_im1, matches_im0, image2.shape[:2], image1_padded.shape[:2])

    aligned_image = mesh_warp_frame(image1_padded, mesh_x, mesh_y, (image2.shape[0], image2.shape[1]))

    matches_im0_from_im1 = mesh_warp_points(matches_im1, mesh_x, mesh_y, (image2.shape[1], image2.shape[0]))
    
    info = compute_img_diff(aligned_image, image2, matches_im0, matches_im0_from_im1, vis=vis)
    
    return aligned_image, info

