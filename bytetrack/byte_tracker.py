import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
from contextlib import nullcontext

from .kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, feat=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        # ReID features
        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=30)
        self.alpha = 0.9

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        # Update features if available
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

    def update_features(self, feat):
        """Update ReID features with exponential moving average."""
        feat = feat / (np.linalg.norm(feat) + 1e-6)  # Normalize
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.smooth_feat = self.smooth_feat / (np.linalg.norm(self.smooth_feat) + 1e-6)
        self.features.append(feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30, enable_reid=False, reid_model=None, reid_model_path=None,
                 reid_threshold=0.5, lambda_emb=0.3, device='cpu', profiler=None):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.profiler = profiler

        # ReID parameters
        self.enable_reid = enable_reid
        self.reid_model = None
        self.reid_threshold = reid_threshold
        self.lambda_emb = lambda_emb
        self.device = device

        if enable_reid:
            if reid_model is not None:
                # Reuse existing model (shared across multiple trackers)
                self.reid_model = reid_model
            elif reid_model_path:
                # Create new model
                from .osnet import build_osnet
                self.reid_model = build_osnet(reid_model_path, pretrained=True, device=device)

    def _profile(self, name: str):
        """Get profiling context manager (no-op if profiler is None)."""
        if self.profiler is not None:
            return self.profiler.profile(name)
        return nullcontext()

    def extract_reid_features(self, frame, detections):
        """Extract ReID features for detections in single frame."""
        if not self.enable_reid or len(detections) == 0 or frame is None:
            return None

        import cv2

        crops = []
        valid_dets = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Resize to 256x128 (OSNet input size)
            crop = cv2.resize(crop, (128, 256))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
            crops.append(crop)
            valid_dets.append(det)

        if len(crops) == 0:
            return None

        # Batch inference
        with torch.no_grad():
            batch = torch.stack(crops).to(self.device)
            features = self.reid_model(batch)  # (N, 512)

        return features.cpu().numpy()

    def update(self, output_results, img_info, img_size, frame=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Step 1: Prepare detections
        with self._profile("1_prepare_detections"):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
            else:
                output_results = output_results.cpu().numpy()
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
            img_h, img_w = img_info[0], img_info[1]
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
            bboxes /= scale

            remain_inds = scores > self.args.track_thresh
            inds_low = scores > 0.1
            inds_high = scores < self.args.track_thresh

            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            scores_second = scores[inds_second]

        # Step 2: Extract ReID features if enabled
        embeddings = None
        embeddings_second = None
        if self.enable_reid and frame is not None:
            with self._profile("2_reid_extraction"):
                if len(dets) > 0:
                    embeddings = self.extract_reid_features(frame, dets)
                if len(dets_second) > 0:
                    embeddings_second = self.extract_reid_features(frame, dets_second)

        # Step 3: Create detection objects
        with self._profile("3_create_detections"):
            if len(dets) > 0:
                '''Detections'''
                if embeddings is not None:
                    detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, feat) for
                                  (tlbr, s, feat) in zip(dets, scores_keep, embeddings)]
                else:
                    detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                  (tlbr, s) in zip(dets, scores_keep)]
            else:
                detections = []

            ''' Add newly detected tracklets to tracked_stracks'''
            unconfirmed = []
            tracked_stracks = []  # type: list[STrack]
            for track in self.tracked_stracks:
                if not track.is_activated:
                    unconfirmed.append(track)
                else:
                    tracked_stracks.append(track)

        ''' Step 4: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        with self._profile("4_kalman_predict"):
            STrack.multi_predict(strack_pool)

        # Compute cost matrix
        with self._profile("5_compute_cost"):
            dists = matching.iou_distance(strack_pool, detections)

            # Fuse with ReID if enabled and features available
            if self.enable_reid and embeddings is not None:
                # Check if tracks have features
                tracks_with_feat = [t for t in strack_pool if t.smooth_feat is not None]
                dets_with_feat = [d for d in detections if d.curr_feat is not None]

                if len(tracks_with_feat) > 0 and len(dets_with_feat) > 0:
                    emb_dists = matching.embedding_distance(tracks_with_feat, dets_with_feat)
                    # Fuse IoU and embedding distance
                    emb_dists = matching.fuse_iou(emb_dists, tracks_with_feat, dets_with_feat)

                    # Map back to full cost matrix
                    track_indices = [strack_pool.index(t) for t in tracks_with_feat]
                    det_indices = [detections.index(d) for d in dets_with_feat]

                    for i, ti in enumerate(track_indices):
                        for j, dj in enumerate(det_indices):
                            # Fuse: lambda_emb * emb_dist + (1 - lambda_emb) * iou_dist
                            dists[ti, dj] = self.lambda_emb * emb_dists[i, j] + (1 - self.lambda_emb) * dists[ti, dj]

            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections)

        # Hungarian assignment
        with self._profile("6_hungarian"):
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        # Update matched tracks
        with self._profile("7_update_matched"):
            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

        ''' Step 5: Second association, with low score detection boxes'''
        with self._profile("8_second_association"):
            # association the untrack to the low score detections
            if len(dets_second) > 0:
                '''Detections'''
                if embeddings_second is not None:
                    detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, feat) for
                                  (tlbr, s, feat) in zip(dets_second, scores_second, embeddings_second)]
                else:
                    detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                  (tlbr, s) in zip(dets_second, scores_second)]
            else:
                detections_second = []
            r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            dists = matching.iou_distance(r_tracked_stracks, detections_second)
            matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections_second[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        with self._profile("9_unconfirmed_tracks"):
            detections = [detections[i] for i in u_detection]
            dists = matching.iou_distance(unconfirmed, detections)
            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
            for itracked, idet in matches:
                unconfirmed[itracked].update(detections[idet], self.frame_id)
                activated_starcks.append(unconfirmed[itracked])
            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)

        """ Step 6: Init new stracks"""
        with self._profile("10_init_new_tracks"):
            for inew in u_detection:
                track = detections[inew]
                if track.score < self.det_thresh:
                    continue
                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(track)
        """ Step 7: Update state"""
        with self._profile("11_update_state"):
            for track in self.lost_stracks:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_stracks.append(track)

            # print('Ramained match {} s'.format(t4-t3))

            self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
            self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
            self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
            self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
            self.lost_stracks.extend(lost_stracks)
            # Only this frame's removals can still be in lost_stracks;
            # earlier removals were subtracted on their own frame. Iterating
            # the full history here (as the original implementation did)
            # makes long runs progressively slower and leaks memory.
            self.lost_stracks = sub_stracks(self.lost_stracks, removed_stracks)
            self.removed_stracks.extend(removed_stracks)
            if len(self.removed_stracks) > 1000:
                self.removed_stracks = self.removed_stracks[-1000:]
            self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
            # get scores of lost tracks
            output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
