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
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL) == EINTR)
      ;
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
  if (pthread_create(&k->thread, NULL, run, k) != 0) {
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
  atomic_store(&k->stop, 1);
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
  free(k);
  return Val_unit;
}
