import copy
import time
from multiprocessing import Process, Queue

import wandb
from nvflare.apis.fl_constant import ProcessType
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.tracking.wandb.wandb_receiver import (
    WandBReceiver,
    WandBTask,
    _check_wandb_args,
)


def _get_job_id_tag(fl_ctx: FLContext) -> str:
    job_id = fl_ctx.get_job_id()
    if job_id == "simulate_job":
        job_id = str(int(time.time()))
    return job_id


class GroupedWandBReceiver(WandBReceiver):
    """W&B receiver that keeps the caller-provided group unchanged.

    NVFlare's default receiver appends an internal job id to the group, which
    makes post-run summary tables hard to group with the client metric runs.
    """

    def initialize(self, fl_ctx: FLContext):
        if fl_ctx.get_process_type() == ProcessType.SERVER_JOB:
            clients = fl_ctx.get_engine().get_clients()
            if not clients:
                raise RuntimeError("No clients found in server context")
            site_names = [c.name for c in clients]
        else:
            site_name = fl_ctx.get_identity_name()
            if not site_name:
                raise RuntimeError("Unable to determine client identity")
            site_names = [site_name]

        self.log_info(fl_ctx, f"Initializing WandB tracking for sites: {site_names}")
        self.fl_ctx = fl_ctx

        run_name = self.wandb_args["name"]
        job_group_name = self.wandb_args.get("group", run_name)
        job_id_tag = _get_job_id_tag(fl_ctx)
        wand_config = self.wandb_args.get("config", {})

        if self.mode == "online":
            try:
                wandb.login(timeout=1, verify=True)
            except Exception as e:
                self.log_warning(fl_ctx, f"Unsuccessful login: {e}. Using wandb offline mode.")
                self.mode = "offline"

        for site_name in site_names:
            self.log_info(fl_ctx, f"initialize WandB run for site {site_name}")
            self.wandb_args["name"] = f"{site_name}-{run_name}"
            self.wandb_args["group"] = job_group_name
            self.wandb_args["mode"] = self.mode
            wand_config["job_id"] = job_id_tag
            wand_config["client"] = site_name
            wand_config["run_name"] = run_name

            _check_wandb_args(self.wandb_args)

            q = Queue()
            q.put(WandBTask(task_owner=site_name, task_type="init", task_data=copy.deepcopy(self.wandb_args), step=0))

            self.queues[site_name] = q
            p = Process(target=self._process_queue_tasks, args=(q,))
            self.processes[site_name] = p
            p.start()
            time.sleep(0.2)
