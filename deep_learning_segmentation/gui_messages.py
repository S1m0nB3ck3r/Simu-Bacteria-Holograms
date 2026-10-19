# -*- coding: utf-8 -*-

"""
Filename: gui_messages.py

Description:
Gestionnaire de messages commun aux GUI du U-Net 3D (GUI_inference.py, GUI_train.py) : le calcul tourne
hors de la boucle Tk, le GUI reste réactif.
  - TaskRunner : lance une fonction de travail dans un fil (mode 'thread' : partage les modèles chargés,
    pour l'inférence) ou dans un processus séparé (mode 'process' : verrou Python propre, le GUI ne
    ralentit jamais, même pendant les boucles Python de l'entraînement ; la fermeture du GUI n'interrompt
    pas une écriture de modèle). La fonction reçoit un Reporter.
  - Reporter (côté calcul) : log(), progress(i, n, texte), frame(données), metrics(dict), stopped() /
    check_stop() (lève Stopped). Messages typés envoyés dans une file.
  - MessagePump (côté GUI) : relève la file par root.after ; au plus budget_ms de traitement par passage
    (une rafale de messages ne gèle pas l'interface) ; les messages 'frame' sont fusionnés : seul le plus
    récent est passé à l'affichage, qui ne prend donc jamais de retard sur le calcul.
Types de messages : 'log', 'progress', 'frame', 'metrics', 'done', 'error', 'stopped'.

Author: Simon BECKER
Date: 2026-10-18

License:
GNU General Public License v3.0
"""

import multiprocessing as mp
import queue
import threading
import time
import traceback


class Stopped(Exception):
    """Levée par Reporter.check_stop() quand l'arrêt a été demandé."""


class Reporter:
    def __init__(self, q, stop_event):
        self.q, self.stop_event = q, stop_event

    def log(self, msg):
        self.q.put(('log', str(msg)))

    def progress(self, i, n, text=''):
        self.q.put(('progress', (int(i), int(n), str(text))))

    def frame(self, data):
        self.q.put(('frame', data))

    def metrics(self, data):
        self.q.put(('metrics', data))

    def stopped(self):
        return self.stop_event.is_set()

    def check_stop(self):
        if self.stop_event.is_set():
            raise Stopped()


def _run(fn, args, q, stop_event):
    rep = Reporter(q, stop_event)
    try:
        q.put(('done', fn(rep, *args)))
    except Stopped:
        q.put(('stopped', None))
    except Exception:
        q.put(('error', traceback.format_exc()))


class TaskRunner:
    """Une tâche à la fois ; mode 'thread' ou 'process' (fn doit alors être importable : fonction de module)."""

    def __init__(self, mode='thread'):
        self.mode = mode
        self.worker = None
        if mode == 'process':
            ctx = mp.get_context('spawn')                 # CUDA : un processus neuf, pas de fork
            self.ctx = ctx
            self.q = ctx.Queue()
            self.stop_event = ctx.Event()
        else:
            self.q = queue.Queue()
            self.stop_event = threading.Event()

    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def start(self, fn, *args):
        if self.busy():
            return False
        self.stop_event.clear()
        if self.mode == 'process':
            self.worker = self.ctx.Process(target=_run, args=(fn, args, self.q, self.stop_event), daemon=False)
        else:
            self.worker = threading.Thread(target=_run, args=(fn, args, self.q, self.stop_event), daemon=True)
        self.worker.start()
        return True

    def stop(self):
        self.stop_event.set()

    def kill(self):
        """Processus seulement : arrêt immédiat (fermeture du GUI pendant une tâche)."""
        if self.mode == 'process' and self.busy():
            self.worker.terminate()


class MessagePump:
    """
    Relève la file du TaskRunner toutes les period_ms et appelle handlers[type](données). Au plus budget_ms
    de traitement par passage ; 'frame' : seul le dernier reçu est affiché (après les autres messages).
    """

    def __init__(self, root, runner, handlers, period_ms=50, budget_ms=30):
        self.root, self.runner, self.handlers = root, runner, handlers
        self.period, self.budget = period_ms, budget_ms / 1000.0
        self._alive = True
        self.root.after(self.period, self._poll)

    def close(self):
        self._alive = False

    def _poll(self):
        if not self._alive:
            return
        t0 = time.perf_counter()
        last_frame = None
        try:
            while time.perf_counter() - t0 < self.budget:
                kind, data = self.runner.q.get_nowait()
                if kind == 'frame':
                    last_frame = data
                    continue
                h = self.handlers.get(kind)
                if h:
                    try:
                        h(data)
                    except Exception:
                        traceback.print_exc()
        except queue.Empty:
            pass
        if last_frame is not None and 'frame' in self.handlers:
            try:
                self.handlers['frame'](last_frame)
            except Exception:
                traceback.print_exc()
        self.root.after(self.period, self._poll)
