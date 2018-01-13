/// <reference path="../../resources/d_ts/phaser.d.ts"/>
var Utils;
(function (Utils) {
    var Timer = (function () {
        function Timer(game) {
            this.game = game;
            this.setStartTime(this.game.time.time);
        }
        Timer.prototype.setStartTime = function (time) { this.startTime = time; };
        Timer.prototype.getStartTime = function () { return this.startTime; };
        Timer.prototype.restart = function () { this.setStartTime(this.game.time.time); };
        Timer.prototype.getCurrentTimeIntSeconds = function () {
            return Utils.toInt(this.game.time.elapsedSecondsSince(this.startTime));
        };
        Timer.prototype.getCurrentTimeSeconds = function () {
            return this.game.time.elapsedSecondsSince(this.startTime);
        };
        Timer.prototype.getCurrentTimeMS = function () {
            return this.game.time.elapsedSince(this.startTime);
        };
        Timer.prototype.getFormattedTime = function () {
            var elapsedSeconds = this.getCurrentTimeIntSeconds();
            var elapsedHours = Utils.toInt(elapsedSeconds / (60 * 60));
            if (elapsedHours > 0) {
                elapsedSeconds -= elapsedHours * 60 * 60;
            }
            var elapsedMinutes = Utils.toInt(elapsedSeconds / 60);
            if (elapsedMinutes > 0) {
                elapsedSeconds -= elapsedMinutes * 60;
            }
            // add 0s for non double digit values
            var retTime = (elapsedHours > 9 ? "" : "0") + elapsedHours + ":" +
                (elapsedMinutes > 9 ? "" : "0") + elapsedMinutes + ":" +
                (elapsedSeconds > 9 ? "" : "0") + elapsedSeconds;
            return retTime;
        };
        return Timer;
    }());
    Utils.Timer = Timer;
})(Utils || (Utils = {}));
//# sourceMappingURL=Timer.js.map