import numpy as np
import math


def rectification(pts):
    
    assert pts.shape == (4, 2)
    homo_pts = np.concatenate([pts, np.ones((4, 1))], axis=1)
    lu, lb, rb, ru = homo_pts
    left_line = np.cross(lu, lb)
    right_line = np.cross(rb, ru)
    
    upper_line = np.cross(lu, ru)
    bottom_line = np.cross(lb, rb)
        
    point_at_inf1 = np.cross(left_line, right_line)
    point_at_inf2 = np.cross(upper_line, bottom_line)
    
    point_at_inf1 = point_at_inf1 / point_at_inf1[2]
    point_at_inf2 = point_at_inf2 / point_at_inf2[2]
    
    line_at_inf = np.cross(point_at_inf1, point_at_inf2)
    l = line_at_inf / line_at_inf[2]
    
    H_rect = np.array([[   1,    0,  0],
                       [   0,    1,  0],
                       [l[0], l[1],  1]])


    # 명함 othogonalize하기
    
    lu, lb, rb, ru = [H_rect @ pt for pt in [lu, lb, rb, ru]]
    
    u1 = np.cross(lu, lb)
    u2 = np.cross(lb, rb)

    v1 = np.cross(lu, rb)
    v2 = np.cross(lb, ru)
    
    A = [[u1[0] * u2[0], u1[0] * u2[1] + u1[1] * u2[0], u1[1] * u2[1]],
         [v1[0] * v2[0], v1[0] * v2[1] + v1[1] * v2[0], v1[1] * v2[1]]]

    U, D, Vh = np.linalg.svd(np.array(A))
    
    x = Vh[-1, :]
    
    K = np.linalg.cholesky(np.array([[x[0], x[1]],
                                     [x[1], x[2]]]))
    
    H_ortho = np.array([[K[0][0], K[0][1], 0],
                        [K[1][0], K[1][1], 0],
                        [      0,       0, 1]])
    H_ortho = np.linalg.inv(H_ortho)
    
    
    lu, lb, rb, ru = [H_ortho @ pt for pt in [lu, lb, rb, ru]]
    
    
    # 명함의 왼쪽 위를 (0, 0)에 위치시키기
    
    t = lu / lu[2]
        
    H_trans = np.array([[1, 0, -t[0]],
                        [0, 1, -t[1]],
                        [0, 0,    1]])
    
    lu, lb, rb, ru = [H_trans @ pt for pt in [lu, lb, rb, ru]]
    lu, lb, rb, ru = [pt / pt[2] for pt in [lu, lb, rb, ru]]
    
    # 명함의 윗변을 x축에 평행하게 회전하기
    
    lu_ru = ru - lu
    
    th = np.arctan2(lu_ru[0], lu_ru[1]) - np.pi * 0.5
    
    
    H_rotate = np.array([[np.cos(th), -np.sin(th), 0],
                         [np.sin(th),  np.cos(th), 0],
                         [         0,           0, 1]])
    
    lu, lb, rb, ru = [H_rotate @ pt for pt in [lu, lb, rb, ru]]
    
    lu, lb, rb, ru = [pt / pt[2] for pt in [lu, lb, rb, ru]]
    
    width_height = (int((ru - lu)[0]), int((lb - lu)[1]))
    
    return H_rotate @ H_trans @ H_ortho @ H_rect, width_height




# https://stackoverflow.com/questions/74843820/compute-aspect-ratio-of-a-rectangle-in-perspective
def compute_aspect_ratio(image, corners):
    # Based on :
    # - https://andrewkay.name/blog/post/aspect-ratio-of-a-rectangle-in-perspective/

    # Step 1: Get image center, will be used as origin
    h, w = image.shape[:2]
    origin = (w * .5, h * .5)
    
    # Step 2: Points coords from image origin
    # /!\ CAREFUL : points need to be in zig-zag order (A, B, D, C)
    a = corners[0] - origin
    b = corners[1] - origin
    c = corners[3] - origin
    d = corners[2] - origin
    
    # Step 3: Check if the camera lie into the plane of the rectangle
    # Coplanar if three points are collinear, in that case the aspect ratio cannot be computed
    M = np.array([[b[0], c[0], d[0]], [b[1], c[1], d[1]], [1., 1., 1.]])
    det = np.linalg.det(M)
    if math.isclose(det, 0., abs_tol=.001):
        # Cannot compute the aspect ratio, the caller need to check if the return value is 0.
        return 0.

    # Step 4: Create the matrixes
    A = np.array([[1., 0., -b[0], 0., 0.,    0.],
                  [0., 1., -b[1], 0., 0.,    0.],
                  [0., 0.,    0., 1., 0., -c[0]],
                  [0., 0.,    0., 0., 1., -c[1]],
                  [1., 0., -d[0], 1., 0., -d[0]],
                  [0., 1., -d[1], 0., 1., -d[1]]], dtype=float)
    
    B = np.array([[b[0]-a[0]],
                  [b[1]-a[1]],
                  [c[0]-a[0]],
                  [c[1]-a[1]],
                  [d[0]-a[0]],
                  [d[1]-a[1]]], dtype=float)

    # Step 5: Solve it, this will give us [Ux, Uy, (Uz / λ), Vx, Vy, (Vz / λ)]
    s = np.linalg.solve(A, B)

    # Step 6: Compute λ, it's the focal length
    l = 0.
    l_sq = ((-(s[0] * s[3]) - (s[1] * s[4])) / (s[2] * s[5]))
    if l_sq > 0.:
        l = np.sqrt(l_sq)
    # If l_sq <= 0, λ cannot be computed, two sides of the rectangle's image are parallel
    # Either Uz and/or Vz is equal zero, so we leave l = 0

    # Step 7: Get U & V
    u = np.linalg.norm([s[0], s[1], (s[2] * l)])
    v = np.linalg.norm([s[3], s[4], (s[5] * l)])

    return (v / u)

