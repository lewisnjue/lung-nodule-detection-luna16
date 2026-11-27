import argparse
import datetime
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from lung_nodule_detection_luna16.dsets import LunaDataset
from lung_nodule_detection_luna16.model import LunaModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s:%(lineno)03d:%(funcName)s %(message)s",
)
log = logging.getLogger(__name__)

METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


class LunaTrainingApp:
    def __init__(self, sys_argv=None) -> None:
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--num-workers",
            help="Number of worker processes for background data loading",
            default=8,
            type=int,
        )
        parser.add_argument(
            "--batch-size",
            help="Batch size to use",
            default=32,
            type=int,
        )
        parser.add_argument(
            "--epochs",
            help="Number of epochs",
            default=1,
            type=int,
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()
        self.totalTrainingSamples_count = 0

    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            log.info(f"Using CUDA; {torch.cuda.device_count()} devices.")
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def initTrainDl(self):
        train_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=False,
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )
        return train_dl

    def initValDl(self):
        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=True,
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )
        return val_dl

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )
        train_progress = tqdm(
            train_dl,
            desc=f"E{epoch_ndx} Training",
            total=len(train_dl),
        )
        for batch_ndx, batch_tup in enumerate(train_progress):
            self.optimizer.zero_grad()
            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g,
            )
            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)
        return trnMetrics_g.to("cpu")

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)
        loss_func = nn.CrossEntropyLoss(reduction="none")
        loss_g = loss_func(
            logits_g,
            label_g[:, 1],
        )

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:, 1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()

        return loss_g.mean()

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )
            val_progress = tqdm(
                val_dl,
                desc=f"E{epoch_ndx} Validation",
                total=len(val_dl),
            )
            for batch_ndx, batch_tup in enumerate(val_progress):
                self.computeBatchLoss(
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    valMetrics_g,
                )

        return valMetrics_g.to("cpu")

    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics_t,
        classificationThreshold=0.5,
    ):
        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        metrics_dict = {}
        metrics_dict["loss/all"] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict["loss/neg"] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict["loss/pos"] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        metrics_dict["correct/all"] = (
            (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) * 100
        )
        metrics_dict["correct/neg"] = neg_correct / np.float32(neg_count) * 100
        metrics_dict["correct/pos"] = pos_correct / np.float32(pos_count) * 100

        log.info(
            f"E{epoch_ndx} {mode_str:8} {metrics_dict['loss/all']:.4f} loss, "
            f"{metrics_dict['correct/all']:-5.1f}% correct"
        )
        log.info(
            f"E{epoch_ndx} {mode_str + '_neg':8} {metrics_dict['loss/neg']:.4f} loss, "
            f"{metrics_dict['correct/neg']:-5.1f}% correct ({neg_correct} of {neg_count})"
        )
        log.info(
            f"E{epoch_ndx} {mode_str + '_pos':8} {metrics_dict['loss/pos']:.4f} loss, "
            f"{metrics_dict['correct/pos']:-5.1f}% correct ({pos_correct} of {pos_count})"
        )

    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, "trn", trnMetrics_t)

            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, "val", valMetrics_t)


if __name__ == "__main__":
    LunaTrainingApp().main()
