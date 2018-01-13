/// <reference path="../../resources/d_ts/phaser.d.ts"/>
module Utils
{
    export class Timer
    {
        game:Phaser.Game;

        startTime: number;

        constructor(game) {this.game = game; this.setStartTime(this.game.time.time);}

        setStartTime(time: number) {this.startTime = time;}

        getStartTime(): number {return this.startTime;}

        restart() {this.setStartTime(this.game.time.time)}

        getCurrentTimeIntSeconds(): number
        {
            return Utils.toInt(this.game.time.elapsedSecondsSince(this.startTime));
        }

        getCurrentTimeSeconds(): number
        {
            return this.game.time.elapsedSecondsSince(this.startTime);
        }

        getCurrentTimeMS():number
        {
            return this.game.time.elapsedSince(this.startTime);
        }

        getFormattedTime():string
        {
            var elapsedSeconds = this.getCurrentTimeIntSeconds();

            var elapsedHours = Utils.toInt(elapsedSeconds / (60 * 60));
            if (elapsedHours > 0)
            {
                elapsedSeconds -= elapsedHours * 60 * 60;
            }
            var elapsedMinutes =  Utils.toInt(elapsedSeconds / 60);
            if (elapsedMinutes > 0)
            {
                elapsedSeconds -= elapsedMinutes * 60;
            }

            // add 0s for non double digit values
            var retTime = (elapsedHours > 9? "" : "0") + elapsedHours + ":" +
                (elapsedMinutes > 9? "" : "0") + elapsedMinutes + ":" +
                (elapsedSeconds > 9? "" : "0") + elapsedSeconds;

            return retTime;
        }

    }
}
