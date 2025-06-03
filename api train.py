from apscheduler.schedulers.background import BackgroundScheduler

def schedule_retraining():
    scheduler = BackgroundScheduler()
    
    @scheduler.scheduled_job('cron', hour=3)  # Daily at 3 AM
    def nightly_retraining():
        new_data = get_new_training_data()
        if new_data:
            train_model(new_data)
    
    scheduler.start()