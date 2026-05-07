use std::sync::{Condvar, Mutex, RwLock};

pub struct SharedState<T> {
    data: RwLock<Option<T>>,
    access_control: Mutex<()>,
    create_event: Condvar,
}

impl<T> SharedState<T> {
    pub fn new() -> Self {
        SharedState {
            data: RwLock::new(None),
            access_control: Mutex::new(()),
            create_event: Condvar::new(),
        }
    }
    pub fn start(&self, data: T) {
        let guard = self.access_control.lock().unwrap();

        *self.data.write().unwrap() = Some(data);
        self.create_event.notify_all();
        drop(guard);
    }

    pub fn stop(&self) {
        let guard = self.access_control.lock().unwrap();
        *self.data.write().unwrap() = None;
        drop(guard);
    }

    pub fn write_op<R>(&self, op: impl FnOnce(&mut T) -> R) -> R {
        let mut guard = self.access_control.lock().unwrap();
        loop {
            let mut write = self.data.write().unwrap();
            match write.as_mut() {
                Some(data) => {
                    drop(guard);
                    return op(data);
                }
                None => {
                    drop(write);
                    guard = self.create_event.wait(guard).unwrap();
                }
            }
        }
    }

    pub fn write_op_if_exists<R>(&self, op: impl FnOnce(&mut T) -> R) -> Option<R> {
        let guard = self.access_control.lock().unwrap();
        let mut write = self.data.write().unwrap();
        drop(guard);
        write.as_mut().map(op)
    }

    pub fn read_op<R>(&self, op: impl FnOnce(&T) -> R) -> R {
        let mut guard = self.access_control.lock().unwrap();
        loop {
            let read = self.data.read().unwrap();
            match read.as_ref() {
                Some(data) => {
                    drop(guard);
                    return op(data);
                }
                None => {
                    drop(read);
                    guard = self.create_event.wait(guard).unwrap();
                }
            }
        }
    }
    pub fn read_op_if_exists<R>(&self, op: impl FnOnce(&T) -> R) -> Option<R> {
        let guard = self.access_control.lock().unwrap();
        let read = self.data.read().unwrap();
        drop(guard);
        read.as_ref().map(op)
    }
}
