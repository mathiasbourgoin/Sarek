/******************************************************************************/
/* SPDX-License-Identifier: CECILL-B                                          */
/* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> */
/******************************************************************************/

/* Link_keepalive: a native thread that enqueues a trivial kernel on a stream
   every period, by calling the driver's launch function directly.

   The thread never touches the OCaml runtime: no callback, no allocation, no
   runtime lock, so it takes part in no stop-the-world collection and is not
   delayed by one. The launch entry point is passed as an address, because
   hipModuleLaunchKernel and cuLaunchKernel take the same arguments in the same
   order: (function, grid x y z, block x y z, shared bytes, stream, params,
   extra). So does [prep], one call made once in the thread before the first
   launch (hipSetDevice with a device index, cuCtxSetCurrent with a context).

   The deadlines are absolute (CLOCK_MONOTONIC), so a slow launch does not make
   the period drift; a thread that fell behind by a whole period restarts from
   now instead of bursting. */

#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

typedef int (*launch_fn_t)(void *, unsigned, unsigned, unsigned, unsigned,
                           unsigned, unsigned, unsigned, void *, void **,
                           void **);
typedef int (*prep_fn_t)(intptr_t);

enum { STATE_STARTING = 0, STATE_RUNNING = 1, STATE_EXITED = 2, STATE_FAILED = 3 };

struct keepalive {
  launch_fn_t launch;
  prep_fn_t prep;
  intptr_t prep_arg;
  void *kernel;
  void *stream;
  long period_ns;
  atomic_int stop;
  /* [paused] and [parked] are guarded by [mutex]; [cond] wakes the thread
     (stop, pause, resume, end of a wait), [ack] wakes a caller waiting for the
     thread to park. */
  int paused;
  int parked;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  pthread_cond_t ack;
  atomic_long launches;
  atomic_int state;
  atomic_int rc;
  pthread_t thread;
  int created;
  int joined;
};

/* A zero-argument kernel reads no parameter, but the driver is given a valid
   array all the same. */
static void *no_params[1] = {NULL};

static void add_ns(struct timespec *t, long ns) {
  t->tv_nsec += ns;
  while (t->tv_nsec >= 1000000000L) {
    t->tv_nsec -= 1000000000L;
    t->tv_sec += 1;
  }
}

static int before(const struct timespec *a, const struct timespec *b) {
  return a->tv_sec < b->tv_sec ||
         (a->tv_sec == b->tv_sec && a->tv_nsec < b->tv_nsec);
}

/* Sleep until [next] (absolute, CLOCK_MONOTONIC), or until a stop or a pause is
   requested: the thread then wakes at once instead of at the next period. */
static void wait_until(struct keepalive *k, const struct timespec *next) {
  pthread_mutex_lock(&k->mutex);
  while (!atomic_load(&k->stop) && !k->paused) {
    if (pthread_cond_timedwait(&k->cond, &k->mutex, next) == ETIMEDOUT) break;
  }
  pthread_mutex_unlock(&k->mutex);
}

/* While paused, park: no launch, no wake-up but the ones that matter. Returns
   with the deadline to restart from if the thread was parked. */
static int park_while_paused(struct keepalive *k) {
  int was_parked = 0;
  pthread_mutex_lock(&k->mutex);
  while (k->paused && !atomic_load(&k->stop)) {
    was_parked = 1;
    k->parked = 1;
    pthread_cond_broadcast(&k->ack);
    pthread_cond_wait(&k->cond, &k->mutex);
  }
  k->parked = 0;
  pthread_mutex_unlock(&k->mutex);
  return was_parked;
}

static void fail(struct keepalive *k, int rc) {
  atomic_store(&k->rc, rc);
  atomic_store(&k->state, STATE_FAILED);
}

static void *run(void *arg) {
  struct keepalive *k = (struct keepalive *)arg;
  sigset_t all;
  sigfillset(&all);
  pthread_sigmask(SIG_BLOCK, &all, NULL);
  int rc = k->prep(k->prep_arg);
  if (rc != 0) {
    fail(k, rc);
    return NULL;
  }
  atomic_store(&k->state, STATE_RUNNING);
  struct timespec next, now;
  clock_gettime(CLOCK_MONOTONIC, &next);
  while (!atomic_load(&k->stop)) {
    if (park_while_paused(k)) {
      clock_gettime(CLOCK_MONOTONIC, &next);
      continue;
    }
    rc = k->launch(k->kernel, 1, 1, 1, 32, 1, 1, 0, k->stream, no_params, NULL);
    if (rc != 0) {
      fail(k, rc);
      return NULL;
    }
    atomic_fetch_add(&k->launches, 1);
    add_ns(&next, k->period_ns);
    clock_gettime(CLOCK_MONOTONIC, &now);
    if (before(&next, &now)) {
      next = now;
      add_ns(&next, k->period_ns);
    }
    wait_until(k, &next);
  }
  atomic_store(&k->state, STATE_EXITED);
  return NULL;
}

/* [args]: launch, prep, prep_arg, kernel, stream (nativeints). Returns the
   handle, or 0 when the thread could not be created. */
CAMLprim value sarek_link_keepalive_start(value args, value period_ns) {
  CAMLparam2(args, period_ns);
  struct keepalive *k = calloc(1, sizeof(struct keepalive));
  if (k == NULL) CAMLreturn(caml_copy_nativeint(0));
  k->launch = (launch_fn_t)Nativeint_val(Field(args, 0));
  k->prep = (prep_fn_t)Nativeint_val(Field(args, 1));
  k->prep_arg = (intptr_t)Nativeint_val(Field(args, 2));
  k->kernel = (void *)Nativeint_val(Field(args, 3));
  k->stream = (void *)Nativeint_val(Field(args, 4));
  k->period_ns = Long_val(period_ns);
  atomic_init(&k->stop, 0);
  atomic_init(&k->launches, 0);
  atomic_init(&k->state, STATE_STARTING);
  atomic_init(&k->rc, 0);
  pthread_condattr_t attr;
  pthread_condattr_init(&attr);
  pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
  pthread_mutex_init(&k->mutex, NULL);
  pthread_cond_init(&k->cond, &attr);
  pthread_cond_init(&k->ack, &attr);
  pthread_condattr_destroy(&attr);
  if (pthread_create(&k->thread, NULL, run, k) != 0) {
    pthread_mutex_destroy(&k->mutex);
    pthread_cond_destroy(&k->cond);
    pthread_cond_destroy(&k->ack);
    free(k);
    CAMLreturn(caml_copy_nativeint(0));
  }
  k->created = 1;
  CAMLreturn(caml_copy_nativeint((intnat)k));
}

CAMLprim value sarek_link_keepalive_state(value handle) {
  struct keepalive *k = (struct keepalive *)Nativeint_val(handle);
  return Val_int(atomic_load(&k->state));
}

CAMLprim value sarek_link_keepalive_rc(value handle) {
  struct keepalive *k = (struct keepalive *)Nativeint_val(handle);
  return Val_int(atomic_load(&k->rc));
}

CAMLprim value sarek_link_keepalive_launches(value handle) {
  struct keepalive *k = (struct keepalive *)Nativeint_val(handle);
  return Val_long(atomic_load(&k->launches));
}

/* Ask the thread to stop and wait for it. The thread notices within one
   period; the runtime lock is released while waiting. Idempotent. */
CAMLprim value sarek_link_keepalive_stop(value handle) {
  struct keepalive *k = (struct keepalive *)Nativeint_val(handle);
  pthread_mutex_lock(&k->mutex);
  atomic_store(&k->stop, 1);
  pthread_cond_broadcast(&k->cond);
  pthread_mutex_unlock(&k->mutex);
  if (k->created && !k->joined) {
    k->joined = 1;
    caml_enter_blocking_section();
    pthread_join(k->thread, NULL);
    caml_leave_blocking_section();
  }
  return Val_unit;
}

/* Only after [stop]. */
CAMLprim value sarek_link_keepalive_free(value handle) {
  struct keepalive *k = (struct keepalive *)Nativeint_val(handle);
  pthread_mutex_destroy(&k->mutex);
  pthread_cond_destroy(&k->cond);
  pthread_cond_destroy(&k->ack);
  free(k);
  return Val_unit;
}

/* Stop launching and return once the thread is parked: no launch is made after
   this returns, until [resume]. The thread, the stream and the kernel stay.
   Idempotent. A thread that has failed or exited is never waited for. */
CAMLprim value sarek_link_keepalive_suspend(value handle) {
  struct keepalive *k = (struct keepalive *)Nativeint_val(handle);
  pthread_mutex_lock(&k->mutex);
  k->paused = 1;
  pthread_cond_broadcast(&k->cond);
  while (!k->parked && atomic_load(&k->state) == STATE_RUNNING &&
         !atomic_load(&k->stop))
    pthread_cond_wait(&k->ack, &k->mutex);
  pthread_mutex_unlock(&k->mutex);
  return Val_unit;
}

CAMLprim value sarek_link_keepalive_resume(value handle) {
  struct keepalive *k = (struct keepalive *)Nativeint_val(handle);
  pthread_mutex_lock(&k->mutex);
  k->paused = 0;
  pthread_cond_broadcast(&k->cond);
  pthread_mutex_unlock(&k->mutex);
  return Val_unit;
}

CAMLprim value sarek_link_keepalive_paused(value handle) {
  struct keepalive *k = (struct keepalive *)Nativeint_val(handle);
  pthread_mutex_lock(&k->mutex);
  int p = k->paused;
  pthread_mutex_unlock(&k->mutex);
  return Val_int(p);
}
