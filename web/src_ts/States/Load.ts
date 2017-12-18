/// <reference path="../../resources/d_ts/phaser.d.ts"/>
module State
{
    export class Load extends Phaser.State
    {
        game: Phaser.Game;

        loadingLabel: Phaser.Text;

        preload()
        {
            // loading text
            this.loadingLabel = this.game.add.text(this.game.world.centerX, this.game.world.centerY, 'loading - 0%',
                {
                    font: '30px Arial',
                    fill: '#ffffff'
                });
            this.loadingLabel.anchor.setTo(0.5, 0.5);
            // progress bar

            let progressBar = this.game.add.sprite(this.game.world.centerX - (0.5 * this.game.cache.getImage('progressBar').width),
                this.game.world.centerY + 30, 'progressBar');
            let progressBarOutline = this.game.add.sprite(this.game.world.centerX - (0.5 * this.game.cache.getImage('progressBarOutline').width),
                this.game.world.centerY + 30, 'progressBarOutline');
            this.game.load.setPreloadSprite(progressBar); // automatically scales progress bar

            /** LOADING MSG **/
            //var text: Phaser.Text = this.game.add.text(10, 10, "Loading...", { font: "48px Arial", fill: "#ff0000", textAlign: "center"});
            //text.anchor.set(0.5,0.5);
            //text.position.set(this.game.width/2,this.game.height/2);


            /** FONTS **/
            //this.game.load.bitmapFont('gamefont', Config.FONTS_PATH+'nokia.png', Config.FONTS_PATH+'nokia.xml'); // load a bitmap font
            // ttf font defined in css/app.css

            /** GFX **/
            this.game.load.image('bg_title', Config.GFX_PATH+'bg_title.png'); // background
            this.game.load.image('bg_menu', Config.GFX_PATH+'bg_menu.png');
            this.game.load.image('bg_game', Config.GFX_PATH+'bg_game.png');
            this.game.load.image('bg_end', Config.GFX_PATH+'bg_end.png');
            this.game.load.image('cc', Config.GFX_PATH+'by-nc-sa.jpg'); // cc logo
            this.game.load.atlasJSONHash("BIRD_FLYING",  Config.SPRITE_SHEETS_PATH+"bird.png",  Config.SPRITE_SHEETS_PATH+"bird_flying.json"); // sprite sheets

            /** AUDIO **/

            this.game.load.audio("music", Config.AUDIO_PATH+"music.mp3"); // music
            this.game.load.audio("ding", Config.SFX_PATH+"ding.mp3"); // sfx

            if (Config.TEST_LOADING)
            {
                // load some assets to slow down loading screen (requires internet)
                let numImages = 10; // increase until you see the progress bar
                for (let i=0;i<numImages;i++)
                    this.game.load.image('dummy'+i, "http://loremflickr.com/"+(320+i)+"/"+(240+i));
            }

        }

        loadUpdate()
        {
            this.loadingLabel.text = 'loading - ' + this.game.load.progress + '%';
        }

        create()
        {

            if (Config.DEBUG)
            {
                var titleButton = this.game.input.keyboard.addKey(Phaser.Keyboard.F1);
                titleButton.onDown.add(() => this.game.state.start("TitleState"));
                var menuButton = this.game.input.keyboard.addKey(Phaser.Keyboard.F2);
                menuButton.onDown.add(() => this.game.state.start("MenuState"));
                var gameButton = this.game.input.keyboard.addKey(Phaser.Keyboard.F3);
                gameButton.onDown.add(() => this.game.state.start("GameState"));
                var endButton = this.game.input.keyboard.addKey(Phaser.Keyboard.F4);
                endButton.onDown.add(() => this.game.state.start("EndState"));

                this.game.input.resetLocked = true; // with this input does not get reset on state change
            }

            this.game.state.start("TitleState");
        }



    }
}