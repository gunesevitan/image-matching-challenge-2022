import logging
import yaml
import numpy as np
import pandas as pd
import torch
from kornia.feature import LoFTR

import settings
import io_utils
import image_utils
import geometry
import loftr
import solver
import metrics
import visualization


if __name__ == '__main__':

    # Read matcher and model configurations
    matcher_config = yaml.load(open(settings.MODELS / 'matcher' / 'config_matcher.yaml', 'r'), Loader=yaml.FullLoader)
    loftr_config = yaml.load(open(settings.MODELS / 'loftr' / 'config_loftr.yaml', 'r'), Loader=yaml.FullLoader)

    if matcher_config['task'] == 'validation':

        # Load pretrained weights for LoFTR model
        device = torch.device(loftr_config['model_parameters']['device'])
        loftr_model = LoFTR(loftr_config['model_parameters']['pretrained'])
        loftr_model = loftr_model.to(device).eval()

        scaling_factors = pd.read_csv(settings.DATA / 'train' / 'scaling_factors.csv').set_index('scene')['scaling_factor'].to_dict()
        mean_average_accuracies = []

        for scene in list(matcher_config['scene_covisibility_threshold'].keys()):

            # Read camera calibration and filtered pair covisibility data for the scene
            df_calibration = io_utils.read_calibration(scene=scene)
            df_pair_covisibility = io_utils.read_pair_covisibility(
                scene=scene,
                n_pairs=matcher_config['validation_pairs'],
                covisibility_threshold=matcher_config['scene_covisibility_threshold'][scene]
            )
            scene_errors = []
            logging.info(f'Evaluating predictions on {scene} - {len(df_pair_covisibility)} pairs {df_pair_covisibility["covisibility"].mean():.4f} average covisibility')

            for idx, row in df_pair_covisibility.iterrows():

                # Get ground-truth camera intrinsics, rotation matrix and translation vector
                image1_id, image2_id = row['pair'].split('-')
                image1_camera_intrinsics, image1_rotation_matrix, image1_translation_vector = geometry.get_camera_calibration_from_dataframe(
                    df_calibration=df_calibration,
                    image_id=image1_id
                )
                image2_camera_intrinsics, image2_rotation_matrix, image2_translation_vector = geometry.get_camera_calibration_from_dataframe(
                    df_calibration=df_calibration,
                    image_id=image2_id
                )
                # Get ground truth fundamental matrix
                ground_truth_fundamental_matrix = geometry.get_fundamental_matrix_from_dataframe(
                    df_pair_covisibility=df_pair_covisibility,
                    pair=f'{image1_id}-{image2_id}'
                )

                # Create image tensors from the image pair and match keypoints using LoFTR model
                image1 = image_utils.get_image_tensor(
                    image=f'{settings.DATA}/train/{scene}/images/{image1_id}.jpg',
                    device=loftr_config['model_parameters']['device'],
                    longest_edge=loftr_config['preprocessing_parameters']['image_longest_edge'],
                    normalize=loftr_config['preprocessing_parameters']['image_normalize'],
                    grayscale=loftr_config['preprocessing_parameters']['image_grayscale']
                )
                image2 = image_utils.get_image_tensor(
                    image=f'{settings.DATA}/train/{scene}/images/{image2_id}.jpg',
                    device=loftr_config['model_parameters']['device'],
                    longest_edge=loftr_config['preprocessing_parameters']['image_longest_edge'],
                    normalize=loftr_config['preprocessing_parameters']['image_normalize'],
                    grayscale=loftr_config['preprocessing_parameters']['image_grayscale']
                )
                loftr_output = loftr.match(
                    image1=image1,
                    image2=image2,
                    loftr_model=loftr_model,
                    confidence_threshold=loftr_config['matching_parameters']['keypoint_confidence_threshold']
                )
                if matcher_config['visualize'] is False:
                    del image1, image2
                    torch.cuda.empty_cache()

                if len(loftr_output['keypoints1']) > matcher_config['solver_parameters']['keypoint_number_threshold']:
                    # Estimate fundamental matrix using predicted keypoints if number of keypoints is more than the specified threshold
                    estimated_fundamental_matrix, inliers = solver.get_fundamental_matrix(
                        keypoints1=loftr_output['keypoints1'],
                        keypoints2=loftr_output['keypoints2'],
                        ransac_reproj_threshold=matcher_config['solver_parameters']['ransac_reproj_threshold'],
                        confidence=matcher_config['solver_parameters']['ransac_confidence'],
                        max_iters=matcher_config['solver_parameters']['ransac_max_iters']
                    )
                else:
                    # Fundamental matrix is 3x3 zeros if number of keypoints is less than the specified threshold
                    estimated_fundamental_matrix = np.zeros((3, 3))
                    inliers = None

                # Decompose estimated fundamental matrix into rotation matrices and translation vector
                estimated_rotation_matrix1, estimated_rotation_matrix2, t_prediction = geometry.decompose_fundamental_matrix_with_camera_intrinsics(
                    fundamental_matrix=estimated_fundamental_matrix,
                    camera_intrinsics1=image1_camera_intrinsics,
                    camera_intrinsics2=image2_camera_intrinsics
                )

                # Prepare ground truth and predictions for error calculation
                q_prediction1 = geometry.rotation_matrix_to_quaternion(estimated_rotation_matrix1)
                q_prediction2 = geometry.rotation_matrix_to_quaternion(estimated_rotation_matrix2)
                r_ground_truth = np.dot(image2_rotation_matrix, image1_rotation_matrix.T)
                t_ground_truth = (image2_translation_vector - np.dot(r_ground_truth, image1_translation_vector)).flatten()
                q_ground_truth = geometry.rotation_matrix_to_quaternion(r_ground_truth)

                # Calculate rotation error in terms of degrees and translation error in terms of meters
                rotation_error1, translation_error1 = metrics.calculate_rotation_and_translation_error(
                    q_ground_truth=q_ground_truth,
                    t_ground_truth=t_ground_truth,
                    q_prediction=q_prediction1,
                    t_prediction=t_prediction,
                    scaling_factor=scaling_factors[scene]
                )
                rotation_error2, translation_error2 = metrics.calculate_rotation_and_translation_error(
                    q_ground_truth=q_ground_truth,
                    t_ground_truth=t_ground_truth,
                    q_prediction=q_prediction2,
                    t_prediction=t_prediction,
                    scaling_factor=scaling_factors[scene]
                )
                rotation_error = min(rotation_error1, rotation_error2)
                translation_error = min(translation_error1, translation_error2)
                scene_errors.append({
                    'pair': row['pair'],
                    'rotation_error': rotation_error,
                    'translation_error': translation_error
                })
                logging.info(f'Pair [{idx + 1 }/{len(df_pair_covisibility)}]: {row["pair"]} ({row["covisibility"]} covisibility) - Rotation Error {rotation_error:.4f} - Translation Error {translation_error:4f}')

                # Visualize matched inlier keypoints for the image pair
                if matcher_config['visualize']:
                    visualization.visualize_matches(
                        image1=image1,
                        image2=image2,
                        keypoints1=loftr_config['keypoints1'],
                        keypoints2=loftr_config['keypoints2'],
                        inliers=inliers
                    )

            # Calculate mean average accuracy from calculated rotation and translation errors
            scene_errors = pd.DataFrame(scene_errors)
            mean_average_accuracy, mean_average_accuracy_rotation, mean_average_accuracy_translation = metrics.calculate_mean_average_accuracy(
                rotation_errors=scene_errors['rotation_error'].values,
                translation_errors=scene_errors['translation_error'].values
            )
            mean_average_accuracies.append({
                'scene': scene,
                'mean_average_accuracy': mean_average_accuracy,
                'mean_average_accuracy_rotation': mean_average_accuracy_rotation,
                'mean_average_accuracy_translation': mean_average_accuracy_translation
            })
            logging.info(f'Scene {scene} - mAA: {mean_average_accuracy:.4f} mAA Rotation: {mean_average_accuracy_rotation:.4f} mAA Translation: {mean_average_accuracy_translation:.4f}')

        mean_average_accuracies = pd.DataFrame(mean_average_accuracies)
        mean_average_accuracies = mean_average_accuracies.mean(axis=0).to_dict()
        logging.info(f'Global Score - mAA {mean_average_accuracies["mean_average_accuracy"]:.4f} mAA Rotation {mean_average_accuracies["mean_average_accuracy_rotation"]:.4f} mAA Translation {mean_average_accuracies["mean_average_accuracy_translation"]:.4f}')

    elif matcher_config['task'] == 'test':

        df = pd.read_csv(settings.DATA / 'test.csv')

        # Load pretrained weights for LoFTR model
        device = torch.device(loftr_config['model_parameters']['device'])
        loftr_model = LoFTR(loftr_config['model_parameters']['pretrained'])
        loftr_model = loftr_model.to(device).eval()

        for idx, row in df.iterrows():

            # Create image tensors from the image pair and match keypoints using LoFTR model
            image1 = image_utils.get_image_tensor(
                image=f'{settings.DATA}/test_images/{row["batch_id"]}/{row["image_1_id"]}.png',
                device=loftr_config['model_parameters']['device'],
                longest_edge=loftr_config['preprocessing_parameters']['image_longest_edge'],
                normalize=loftr_config['preprocessing_parameters']['image_normalize'],
                grayscale=loftr_config['preprocessing_parameters']['image_grayscale']

            )
            image2 = image_utils.get_image_tensor(
                image=f'{settings.DATA}/test_images/{row["batch_id"]}/{row["image_2_id"]}.png',
                device=loftr_config['model_parameters']['device'],
                longest_edge=loftr_config['preprocessing_parameters']['image_longest_edge'],
                normalize=loftr_config['preprocessing_parameters']['image_normalize'],
                grayscale=loftr_config['preprocessing_parameters']['image_grayscale']
            )
            loftr_output = loftr.match(
                image1=image1,
                image2=image2,
                loftr_model=loftr_model,
                confidence_threshold=loftr_config['matching_parameters']['keypoint_confidence_threshold']
            )
            if matcher_config['visualize'] is False:
                del image1, image2
                torch.cuda.empty_cache()

            if len(loftr_output['keypoints1']) > matcher_config['solver_parameters']['keypoint_number_threshold']:
                # Estimate fundamental matrix using predicted keypoints if number of keypoints is more than the specified threshold
                estimated_fundamental_matrix, inliers = solver.get_fundamental_matrix(
                    keypoints1=loftr_output['keypoints1'],
                    keypoints2=loftr_output['keypoints2'],
                    ransac_reproj_threshold=matcher_config['solver_parameters']['ransac_reproj_threshold'],
                    confidence=matcher_config['solver_parameters']['ransac_confidence'],
                    max_iters=matcher_config['solver_parameters']['ransac_max_iters']
                )
            else:
                # Fundamental matrix is 3x3 zeros if number of keypoints is less than the specified threshold
                estimated_fundamental_matrix = np.zeros((3, 3))
                inliers = None

            df.loc[idx, 'fundamental_matrix'] = ' '.join([f'{x:.{8}e}' for x in estimated_fundamental_matrix.flatten()])

            # Visualize matched inlier keypoints for the image pair
            if matcher_config['visualize']:
                visualization.visualize_matches(
                    image1=image1,
                    image2=image2,
                    keypoints1=loftr_output['keypoints1'],
                    keypoints2=loftr_output['keypoints2'],
                    inliers=inliers
                )
