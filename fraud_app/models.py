from django.db import models
class PredictionHistory(models.Model):
    result=models.CharField(max_length=20)
    fraud_probability=models.FloatField()
    risk_score=models.IntegerField()
    risk_level=models.CharField(max_length=20)
    created_at=models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return f"{self.result}-{self.risk_score}"
