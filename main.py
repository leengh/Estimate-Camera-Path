from glob import glob
import cv2
import skimage
import os
import numpy as np

class OdometryClass:
    def __init__(self, frame_path):
        self.frame_path = frame_path
        self.frames = sorted(
            glob(os.path.join(self.frame_path, 'images', '*')))
        with open(os.path.join(frame_path, 'calib.txt'), 'r') as f:
            lines = f.readlines()
            self.focal_length = float(lines[0].strip().split()[-1])
            lines[1] = lines[1].strip().split()
            self.pp = (float(lines[1][1]), float(lines[1][2]))

        with open(os.path.join(self.frame_path, 'gt_sequence.txt'), 'r') as f:
            self.pose = [line.strip().split() for line in f.readlines()]
        self.K = np.array([[self.focal_length, 0, self.pp[0]], [
            0, self.focal_length, self.pp[1]], [0, 0, 1]])

    def imread(self, fname):
        """
        read image into np array from file
        """
        return cv2.imread(fname, 0)

    def imshow(self, img):
        """
        show image
        """
        skimage.io.imshow(img)

    def get_gt(self, frame_id):
        pose = self.pose[frame_id]
        x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        return np.array([[x], [y], [z]])

    def get_scale(self, frame_id):
        '''Provides scale estimation for mutliplying
        translation vectors

        Returns:
        float -- Scalar value allowing for scale estimation
        '''
        prev_coords = self.get_gt(frame_id - 1)
        curr_coords = self.get_gt(frame_id)
        return np.linalg.norm(curr_coords - prev_coords)

    def get_keypoints(self, img):
        fast = cv2.FastFeatureDetector_create(
            threshold=25, nonmaxSuppression=True)
        kp = fast.detect(img)
        return np.array([x.pt for x in kp], dtype=np.float32)

    def feature_tracking(self, img1, img2, points1):
        lk_params = dict(winSize=(8, 8),
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        points2, status, err = cv2.calcOpticalFlowPyrLK(
            img1, img2, points1, None, **lk_params)
        status = status.reshape(status.shape[0])
        points1 = points1[status == 1]
        points2 = points2[status == 1]
        return points1, points2

    def decompose_essential_matrix(self, E):
        [U, _, VT] = np.linalg.svd(E)
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        t = U[:, 2]
        R1 = U @ W @ VT
        R2 = U @ W.T @ VT

        K_sign = np.sign(np.linalg.det(self.K))
        R1 = R1 * np.sign(np.linalg.det(R1)) * K_sign
        R2 = R2 * np.sign(np.linalg.det(R2)) * K_sign

        return [[R1, -t], [R1, t], [R2, -t], [R2, t]]

    def triangulate_points(self, P1, P2, kpts1, kpts2):
        X = []
        for (x1, y1), (x2, y2) in zip(kpts1, kpts2):
            A1 = np.array([[0, -1, y1], [1, 0, -x1]])
            A2 = np.array([[0, -1, y2], [1, 0, -x2]])
            A = np.vstack((A1 @ P1, A2 @ P2))
            U, S, V = np.linalg.svd(np.array(A))
            X1 = V[-1:, :].reshape(1, -1)
            X.append(X1)
        X = np.vstack(X)
        return X

    def calculate_relative_scale(self, R, t, kpts1, kpts2):
        zeros = np.zeros((3, 1), dtype=int)
        P1 = np.hstack((self.K, zeros))

        T = np.hstack((R, t.reshape(3, 1)))
        T = np.vstack((T, [0, 0, 0, 1]))

        P2 = np.matmul(np.concatenate((self.K, zeros), axis=1), T)

        triangulate_points1 = self.triangulate_points(P1, P2, kpts1, kpts2)
        triangulate_points1 = triangulate_points1.T

        triangulate_points2 = np.matmul(T, triangulate_points1)

        triangulate_points1_1 = triangulate_points1[:3, :] / triangulate_points1[3, :]
        triangulate_points2_2 = triangulate_points2[:3, :] / triangulate_points2[3, :]

        total_pos = np.sum(triangulate_points1_1[2, :] > 0) + np.sum(triangulate_points2_2[2, :] > 0)

        norm1 = np.linalg.norm(triangulate_points1_1.T[:-1] - triangulate_points1_1.T[1:], axis=-1)
        norm2 = np.linalg.norm(triangulate_points2_2.T[:-1] - triangulate_points2_2.T[1:], axis=-1)
        relative_scale = np.mean(norm1 / norm2)

        return total_pos, relative_scale

    def get_rotation_and_trans(self, E, kpts1, kpts2):

        pairs = self.decompose_essential_matrix(E)

        positive_z = []
        relative_scales = []
        for pair in pairs:
            R, t = pair
            total_pos, relative_scale = self.calculate_relative_scale(
                R, t, kpts1, kpts2)
            positive_z.append(total_pos)
            relative_scales.append(relative_scale)

        max_index = np.argmax(positive_z)
        R, t = pairs[max_index]
        relative_scale = relative_scales[max_index]

        return R, t * relative_scale

    def run(self):
        """
        Uses the video frame to predict the path taken by the camera

        The returned path should be a numpy array of shape nx3
        n is the number of frames in the video
        """
        n = len(self.frames)
        min_features = 2000
        path = np.ones((n, 3))
        self.R = np.eye(3)
        self.t = np.zeros((3, 1)).squeeze()

        img1 = self.imread(self.frames[0])
        points_ref = self.get_keypoints(img1)

        for index in range(1, n):
            prev_frame = self.imread(self.frames[index - 1])
            current_frame = self.imread(self.frames[index])
            kpts1, kpts2 = self.feature_tracking(
                prev_frame, current_frame, points_ref)

            E, _ = cv2.findEssentialMat(
                kpts2,
                kpts1,
                self.focal_length,
                self.pp,
                cv2.FM_8POINT,
                0.999,
                1.0,
                None,
            )

            R, t = self.get_rotation_and_trans(E, kpts2, kpts1)

            scale = self.get_scale(index)
            self.t = self.t + scale * self.R.dot(t)
            self.R = R.dot(self.R)

            if kpts2.shape[0] < min_features:
                kpts2 = self.get_keypoints(current_frame)

            points_ref = kpts2
            point = self.t.copy()
            point[1] *= -1
            path[index, :] = point

        return path

if __name__ == "__main__":
    odometry = OdometryClass(frame_path='video_train')
    path = odometry.run()
    print("path", path)